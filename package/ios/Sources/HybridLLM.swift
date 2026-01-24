import Foundation
import NitroModules
internal import MLX
internal import MLXLLM
internal import MLXLMCommon
internal import Tokenizers

class HybridLLM: HybridLLMSpec {
    private var session: ChatSession?
    private var currentTask: Task<String, Error>?
    private var container: Any?
    private var lastStats: GenerationStats = GenerationStats(
        tokenCount: 0,
        tokensPerSecond: 0,
        timeToFirstToken: 0,
        totalTime: 0,
        toolExecutionTime: 0
    )
    private var modelFactory: ModelFactory = LLMModelFactory.shared
    private var manageHistory: Bool = false
    private var messageHistory: [LLMMessage] = []
    private var loadTask: Task<Void, Error>?

    private var tools: [ToolDefinition] = []
    private var toolSchemas: [ToolSpec] = []

    var isLoaded: Bool { session != nil }
    var isGenerating: Bool { currentTask != nil }
    var modelId: String = ""
    var debug: Bool = false
    var systemPrompt: String = "You are a helpful assistant."
    var additionalContext: LLMMessage = LLMMessage()

    private func log(_ message: String) {
        if debug {
            print("[MLXReactNative.HybridLLM] \(message)")
        }
    }

    private func getMemoryUsage() -> String {
        var taskInfo = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        let result: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }

        if result == KERN_SUCCESS {
            let usedMB = Float(taskInfo.resident_size) / 1024.0 / 1024.0
            return String(format: "%.1f MB", usedMB)
        } else {
            return "unknown"
        }
    }

    private func getGPUMemoryUsage() -> String {
        let snapshot = GPU.snapshot()
        let allocatedMB = Float(snapshot.activeMemory) / 1024.0 / 1024.0
        let cacheMB = Float(snapshot.cacheMemory) / 1024.0 / 1024.0
        let peakMB = Float(snapshot.peakMemory) / 1024.0 / 1024.0
        return String(format: "Allocated: %.1f MB, Cache: %.1f MB, Peak: %.1f MB",
                     allocatedMB, cacheMB, peakMB)
    }

    private func buildToolSchema(from tool: ToolDefinition) -> ToolSpec {
        var properties: [String: [String: Any]] = [:]
        var required: [String] = []

        for param in tool.parameters {
            properties[param.name] = [
                "type": param.type,
                "description": param.description
            ]
            if param.required {
                required.append(param.name)
            }
        }

        return [
            "type": "function",
            "function": [
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    "type": "object",
                    "properties": properties,
                    "required": required
                ]
            ]
        ] as ToolSpec
    }

    func load(modelId: String, options: LLMLoadOptions?) throws -> Promise<Void> {
        self.loadTask?.cancel()

        return Promise.async { [self] in
            let task = Task { @MainActor in
                MLX.GPU.set(cacheLimit: 2000000)

                self.currentTask?.cancel()
                self.currentTask = nil
                self.session = nil
                self.container = nil
                self.tools = []
                self.toolSchemas = []
                MLX.GPU.clearCache()

                let memoryAfterCleanup = self.getMemoryUsage()
                let gpuAfterCleanup = self.getGPUMemoryUsage()
                log("After cleanup - Host: \(memoryAfterCleanup), GPU: \(gpuAfterCleanup)")

                let modelDir = await ModelDownloader.shared.getModelDirectory(modelId: modelId)
                log("Loading from directory: \(modelDir.path)")

                let config = ModelConfiguration(directory: modelDir)
                let loadedContainer = try await self.modelFactory.loadContainer(
                    configuration: config
                ) { progress in
                    options?.onProgress?(progress.fractionCompleted)
                }

                try Task.checkCancellation()

                let memoryAfterContainer = self.getMemoryUsage()
                let gpuAfterContainer = self.getGPUMemoryUsage()
                log("Model loaded - Host: \(memoryAfterContainer), GPU: \(gpuAfterContainer)")

                if let jsTools = options?.tools {
                    self.tools = jsTools
                    self.toolSchemas = jsTools.map { self.buildToolSchema(from: $0) }
                    log("Loaded \(self.tools.count) tools: \(self.tools.map { $0.name })")
                }

                let additionalContextDict: [String: Any]? = if let messages = options?.additionalContext {
                    ["messages": messages.map { ["role": $0.role, "content": $0.content] }]
                } else {
                    nil
                }

                self.container = loadedContainer
                self.session = ChatSession(loadedContainer, instructions: self.systemPrompt, additionalContext: additionalContextDict)
                self.modelId = modelId

                self.manageHistory = options?.manageHistory ?? false
                self.messageHistory = options?.additionalContext ?? []

                if self.manageHistory {
                    log("History management enabled with \(self.messageHistory.count) initial messages")
                }
            }

            self.loadTask = task
            try await task.value
        }
    }

    func generate(prompt: String) throws -> Promise<String> {
        guard let session = session else {
            throw LLMError.notLoaded
        }

        return Promise.async { [self] in
            if self.manageHistory {
                self.messageHistory.append(LLMMessage(role: "user", content: prompt))
            }

            let task = Task<String, Error> {
                log("Generating response for: \(prompt.prefix(50))...")
                let result = try await session.respond(to: prompt)
                log("Generation complete")
                return result
            }

            self.currentTask = task

            do {
                let result = try await task.value
                self.currentTask = nil

                if self.manageHistory {
                    self.messageHistory.append(LLMMessage(role: "assistant", content: result))
                }

                return result
            } catch {
                self.currentTask = nil
                throw error
            }
        }
    }

    private let maxToolCallDepth = 10

    func stream(
        prompt: String,
        onToken: @escaping (String) -> Void,
        onToolCall: ((String, String) -> Void)?
    ) throws -> Promise<String> {
        guard let container = container as? ModelContainer else {
            throw LLMError.notLoaded
        }

        return Promise.async { [self] in
            if self.manageHistory {
                self.messageHistory.append(LLMMessage(role: "user", content: prompt))
            }

            let task = Task<String, Error> {
                let startTime = Date()
                var firstTokenTime: Date?
                var tokenCount = 0

                let result = try await self.performGeneration(
                    container: container,
                    prompt: prompt,
                    toolResults: nil,
                    depth: 0,
                    onToken: { token in
                        if firstTokenTime == nil {
                            firstTokenTime = Date()
                        }
                        tokenCount += 1
                        onToken(token)
                    },
                    onToolCall: onToolCall ?? { _, _ in }
                )

                let endTime = Date()
                let totalTime = endTime.timeIntervalSince(startTime) * 1000
                let timeToFirstToken = (firstTokenTime ?? endTime).timeIntervalSince(startTime) * 1000
                let tokensPerSecond = totalTime > 0 ? Double(tokenCount) / (totalTime / 1000) : 0

                self.lastStats = GenerationStats(
                    tokenCount: Double(tokenCount),
                    tokensPerSecond: tokensPerSecond,
                    timeToFirstToken: timeToFirstToken,
                    totalTime: totalTime,
                    toolExecutionTime: 0
                )

                log("Stream complete - \(tokenCount) tokens, \(String(format: "%.1f", tokensPerSecond)) tokens/s")
                return result
            }

            self.currentTask = task

            do {
                let result = try await task.value
                self.currentTask = nil

                if self.manageHistory {
                    self.messageHistory.append(LLMMessage(role: "assistant", content: result))
                }

                return result
            } catch {
                self.currentTask = nil
                throw error
            }
        }
    }

    func streamWithEvents(
        prompt: String,
        onEvent: @escaping (String) -> Void
    ) throws -> Promise<String> {
        guard let container = container as? ModelContainer else {
            throw LLMError.notLoaded
        }

        return Promise.async { [self] in
            if self.manageHistory {
                self.messageHistory.append(LLMMessage(role: "user", content: prompt))
            }

            let task = Task<String, Error> {
                let startTime = Date()
                var firstTokenTime: Date?
                var outputTokenCount = 0
                var mlxTokenCount = 0
                var mlxGenerationTime: Double = 0
                var toolExecutionTime: Double = 0
                let emitter = StreamEventEmitter(callback: onEvent)

                emitter.emitGenerationStart()

                let result = try await self.performGenerationWithEvents(
                    container: container,
                    prompt: prompt,
                    toolResults: nil,
                    depth: 0,
                    emitter: emitter,
                    onTokenProcessed: {
                        if firstTokenTime == nil {
                            firstTokenTime = Date()
                        }
                        outputTokenCount += 1
                    },
                    onGenerationInfo: { tokens, time in
                        mlxTokenCount += tokens
                        mlxGenerationTime += time
                    },
                    toolExecutionTime: &toolExecutionTime
                )

                let endTime = Date()
                let totalTime = endTime.timeIntervalSince(startTime) * 1000
                let timeToFirstToken = (firstTokenTime ?? endTime).timeIntervalSince(startTime) * 1000
                let tokensPerSecond = mlxGenerationTime > 0 ? Double(mlxTokenCount) / (mlxGenerationTime / 1000) : 0

                let stats = GenerationStats(
                    tokenCount: Double(mlxTokenCount),
                    tokensPerSecond: tokensPerSecond,
                    timeToFirstToken: timeToFirstToken,
                    totalTime: totalTime,
                    toolExecutionTime: toolExecutionTime
                )

                self.lastStats = stats
                emitter.emitGenerationEnd(content: result, stats: stats)

                log("StreamWithEvents complete - \(mlxTokenCount) tokens, \(String(format: "%.1f", tokensPerSecond)) tokens/s (tool execution: \(String(format: "%.0f", toolExecutionTime))ms)")
                return result
            }

            self.currentTask = task

            do {
                let result = try await task.value
                self.currentTask = nil

                if self.manageHistory {
                    self.messageHistory.append(LLMMessage(role: "assistant", content: result))
                }

                return result
            } catch {
                self.currentTask = nil
                throw error
            }
        }
    }

    private func performGenerationWithEvents(
        container: ModelContainer,
        prompt: String,
        toolResults: [String]?,
        depth: Int,
        emitter: StreamEventEmitter,
        onTokenProcessed: @escaping () -> Void,
        onGenerationInfo: @escaping (Int, Double) -> Void,
        toolExecutionTime: inout Double
    ) async throws -> String {
        if depth >= maxToolCallDepth {
            log("Max tool call depth reached (\(maxToolCallDepth))")
            return ""
        }

        var output = ""
        let thinkingMachine = ThinkingStateMachine()
        var pendingToolCalls: [(id: String, tool: ToolDefinition, args: [String: Any], argsJson: String)] = []

        var chat: [Chat.Message] = []

        if !self.systemPrompt.isEmpty {
            chat.append(.system(self.systemPrompt))
        }

        for msg in self.messageHistory {
            switch msg.role {
            case "user": chat.append(.user(msg.content))
            case "assistant": chat.append(.assistant(msg.content))
            case "system": chat.append(.system(msg.content))
            case "tool": chat.append(.tool(msg.content))
            default: break
            }
        }

        if depth == 0 {
            chat.append(.user(prompt))
        }

        if let toolResults = toolResults {
            for result in toolResults {
                chat.append(.tool(result))
            }
        }

        let userInput = UserInput(
            chat: chat,
            tools: !self.toolSchemas.isEmpty ? self.toolSchemas : nil
        )

        let lmInput = try await container.prepare(input: userInput)

        let stream = try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 2048, temperature: 0.7)
            return try MLXLMCommon.generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )
        }

        for await generation in stream {
            if Task.isCancelled { break }

            switch generation {
            case .chunk(let text):
                let outputs = thinkingMachine.process(token: text)

                for machineOutput in outputs {
                    switch machineOutput {
                    case .token(let token):
                        output += token
                        emitter.emitToken(token)
                        onTokenProcessed()

                    case .thinkingStart:
                        emitter.emitThinkingStart()

                    case .thinkingChunk(let chunk):
                        emitter.emitThinkingChunk(chunk)

                    case .thinkingEnd(let content):
                        emitter.emitThinkingEnd(content)
                    }
                }

            case .toolCall(let toolCall):
                log("Tool call detected: \(toolCall.function.name)")

                guard let tool = self.tools.first(where: { $0.name == toolCall.function.name }) else {
                    log("Unknown tool: \(toolCall.function.name)")
                    continue
                }

                let toolCallId = UUID().uuidString
                let argsDict = self.convertToolCallArguments(toolCall.function.arguments)
                let argsJson = dictionaryToJson(argsDict)

                emitter.emitToolCallStart(id: toolCallId, name: toolCall.function.name, arguments: argsJson)
                pendingToolCalls.append((id: toolCallId, tool: tool, args: argsDict, argsJson: argsJson))

            case .info(let info):
                log("Generation info: \(info.generationTokenCount) tokens, \(String(format: "%.1f", info.tokensPerSecond)) tokens/s")
                let generationTime = info.tokensPerSecond > 0 ? Double(info.generationTokenCount) / info.tokensPerSecond * 1000 : 0
                onGenerationInfo(info.generationTokenCount, generationTime)
            }
        }

        let flushOutputs = thinkingMachine.flush()
        for machineOutput in flushOutputs {
            switch machineOutput {
            case .token(let token):
                output += token
                emitter.emitToken(token)
                onTokenProcessed()
            case .thinkingStart:
                emitter.emitThinkingStart()
            case .thinkingChunk(let chunk):
                emitter.emitThinkingChunk(chunk)
            case .thinkingEnd(let content):
                emitter.emitThinkingEnd(content)
            }
        }

        if !pendingToolCalls.isEmpty {
            log("Executing \(pendingToolCalls.count) tool call(s)")

            let toolStartTime = Date()
            var allToolResults: [String] = []

            for (id, tool, argsDict, _) in pendingToolCalls {
                emitter.emitToolCallExecuting(id: id)

                do {
                    let argsAnyMap = self.dictionaryToAnyMap(argsDict)
                    let outerPromise = tool.handler(argsAnyMap)
                    let innerPromise = try await outerPromise.await()
                    let resultAnyMap = try await innerPromise.await()
                    let resultDict = self.anyMapToDictionary(resultAnyMap)
                    let resultJson = dictionaryToJson(resultDict)

                    log("Tool result for \(tool.name): \(resultJson.prefix(100))...")
                    emitter.emitToolCallCompleted(id: id, result: resultJson)
                    allToolResults.append(resultJson)
                } catch {
                    log("Tool execution error for \(tool.name): \(error)")
                    emitter.emitToolCallFailed(id: id, error: error.localizedDescription)
                    allToolResults.append("{\"error\": \"Tool execution failed\"}")
                }
            }

            toolExecutionTime += Date().timeIntervalSince(toolStartTime) * 1000

            if !output.isEmpty {
                self.messageHistory.append(LLMMessage(role: "assistant", content: output))
            }

            for result in allToolResults {
                self.messageHistory.append(LLMMessage(role: "tool", content: result))
            }

            let continuation = try await self.performGenerationWithEvents(
                container: container,
                prompt: prompt,
                toolResults: allToolResults,
                depth: depth + 1,
                emitter: emitter,
                onTokenProcessed: onTokenProcessed,
                onGenerationInfo: onGenerationInfo,
                toolExecutionTime: &toolExecutionTime
            )

            return output + continuation
        }

        return output
    }

    private func performGeneration(
        container: ModelContainer,
        prompt: String,
        toolResults: [String]?,
        depth: Int,
        onToken: @escaping (String) -> Void,
        onToolCall: @escaping (String, String) -> Void
    ) async throws -> String {
        if depth >= maxToolCallDepth {
            log("Max tool call depth reached (\(maxToolCallDepth))")
            return ""
        }

        var output = ""
        var pendingToolCalls: [(tool: ToolDefinition, args: [String: Any], argsJson: String)] = []

        var chat: [Chat.Message] = []

        if !self.systemPrompt.isEmpty {
            chat.append(.system(self.systemPrompt))
        }

        for msg in self.messageHistory {
            switch msg.role {
            case "user": chat.append(.user(msg.content))
            case "assistant": chat.append(.assistant(msg.content))
            case "system": chat.append(.system(msg.content))
            case "tool": chat.append(.tool(msg.content))
            default: break
            }
        }

        if depth == 0 {
            chat.append(.user(prompt))
        }

        if let toolResults = toolResults {
            for result in toolResults {
                chat.append(.tool(result))
            }
        }

        let userInput = UserInput(
            chat: chat,
            tools: !self.toolSchemas.isEmpty ? self.toolSchemas : nil
        )

        let lmInput = try await container.prepare(input: userInput)

        let stream = try await container.perform { context in
            let parameters = GenerateParameters(maxTokens: 2048, temperature: 0.7)
            return try MLXLMCommon.generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )
        }

        for await generation in stream {
            if Task.isCancelled { break }

            switch generation {
            case .chunk(let text):
                output += text
                onToken(text)

            case .toolCall(let toolCall):
                log("Tool call detected: \(toolCall.function.name)")

                guard let tool = self.tools.first(where: { $0.name == toolCall.function.name }) else {
                    log("Unknown tool: \(toolCall.function.name)")
                    continue
                }

                let argsDict = self.convertToolCallArguments(toolCall.function.arguments)
                let argsJson = dictionaryToJson(argsDict)

                pendingToolCalls.append((tool: tool, args: argsDict, argsJson: argsJson))
                onToolCall(toolCall.function.name, argsJson)

            case .info(let info):
                log("Generation info: \(info.generationTokenCount) tokens, \(String(format: "%.1f", info.tokensPerSecond)) tokens/s")
            }
        }

        if !pendingToolCalls.isEmpty {
            log("Executing \(pendingToolCalls.count) tool call(s)")

            var allToolResults: [String] = []

            for (tool, argsDict, _) in pendingToolCalls {
                do {
                    let argsAnyMap = self.dictionaryToAnyMap(argsDict)
                    let outerPromise = tool.handler(argsAnyMap)
                    let innerPromise = try await outerPromise.await()
                    let resultAnyMap = try await innerPromise.await()
                    let resultDict = self.anyMapToDictionary(resultAnyMap)
                    let resultJson = dictionaryToJson(resultDict)

                    log("Tool result for \(tool.name): \(resultJson.prefix(100))...")
                    allToolResults.append(resultJson)
                } catch {
                    log("Tool execution error for \(tool.name): \(error)")
                    allToolResults.append("{\"error\": \"Tool execution failed\"}")
                }
            }

            if !output.isEmpty {
                self.messageHistory.append(LLMMessage(role: "assistant", content: output))
            }

            if depth == 0 {
                self.messageHistory.append(LLMMessage(role: "user", content: prompt))
            }

            for result in allToolResults {
                self.messageHistory.append(LLMMessage(role: "tool", content: result))
            }

            onToken("\u{200B}")

            let continuation = try await self.performGeneration(
                container: container,
                prompt: prompt,
                toolResults: allToolResults,
                depth: depth + 1,
                onToken: onToken,
                onToolCall: onToolCall
            )

            return output + continuation
        }

        return output
    }

    private func convertToolCallArguments(_ arguments: [String: JSONValue]) -> [String: Any] {
        var result: [String: Any] = [:]
        for (key, value) in arguments {
            result[key] = value.anyValue
        }
        return result
    }

    private func dictionaryToAnyMap(_ dict: [String: Any]) -> AnyMap {
        let anyMap = AnyMap()
        for (key, value) in dict {
            switch value {
            case let stringValue as String:
                anyMap.setString(key: key, value: stringValue)
            case let doubleValue as Double:
                anyMap.setDouble(key: key, value: doubleValue)
            case let intValue as Int:
                anyMap.setDouble(key: key, value: Double(intValue))
            case let boolValue as Bool:
                anyMap.setBoolean(key: key, value: boolValue)
            default:
                anyMap.setString(key: key, value: String(describing: value))
            }
        }
        return anyMap
    }

    private func anyMapToDictionary(_ anyMap: AnyMap) -> [String: Any] {
        var dict: [String: Any] = [:]
        for key in anyMap.getAllKeys() {
            if anyMap.isString(key: key) {
                dict[key] = anyMap.getString(key: key)
            } else if anyMap.isDouble(key: key) {
                dict[key] = anyMap.getDouble(key: key)
            } else if anyMap.isBool(key: key) {
                dict[key] = anyMap.getBoolean(key: key)
            }
        }
        return dict
    }

    func stop() throws {
        currentTask?.cancel()
        currentTask = nil
    }

    func unload() throws {
        loadTask?.cancel()
        loadTask = nil

        let memoryBefore = getMemoryUsage()
        let gpuBefore = getGPUMemoryUsage()
        log("Before unload - Host: \(memoryBefore), GPU: \(gpuBefore)")

        currentTask?.cancel()
        currentTask = nil
        session = nil
        container = nil
        tools = []
        toolSchemas = []
        messageHistory = []
        manageHistory = false
        modelId = ""

        MLX.GPU.clearCache()

        let memoryAfter = getMemoryUsage()
        let gpuAfter = getGPUMemoryUsage()
        log("After unload - Host: \(memoryAfter), GPU: \(gpuAfter)")
    }

    func getLastGenerationStats() throws -> GenerationStats {
        return lastStats
    }

    func getHistory() throws -> [LLMMessage] {
        return messageHistory
    }

    func clearHistory() throws {
        messageHistory = []
        log("Message history cleared")
    }
}
