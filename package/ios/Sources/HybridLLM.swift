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
        totalTime: 0
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

    func stream(prompt: String, onToken: @escaping (String) -> Void) throws -> Promise<String> {
        guard let session = session else {
            throw LLMError.notLoaded
        }

        return Promise.async { [self] in
            if self.manageHistory {
                self.messageHistory.append(LLMMessage(role: "user", content: prompt))
            }

            let task = Task<String, Error> {
                var result = ""
                var tokenCount = 0
                let startTime = Date()
                var firstTokenTime: Date?

                log("Streaming response for: \(prompt.prefix(50))...")
                for try await chunk in session.streamResponse(to: prompt) {
                    if Task.isCancelled { break }

                    if firstTokenTime == nil {
                        firstTokenTime = Date()
                    }
                    tokenCount += 1
                    result += chunk
                    onToken(chunk)
                }

                let endTime = Date()
                let totalTime = endTime.timeIntervalSince(startTime) * 1000
                let timeToFirstToken = (firstTokenTime ?? endTime).timeIntervalSince(startTime) * 1000
                let tokensPerSecond = totalTime > 0 ? Double(tokenCount) / (totalTime / 1000) : 0

                self.lastStats = GenerationStats(
                    tokenCount: Double(tokenCount),
                    tokensPerSecond: tokensPerSecond,
                    timeToFirstToken: timeToFirstToken,
                    totalTime: totalTime
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

    private let maxToolCallDepth = 10

    func streamWithTools(
        prompt: String,
        onToken: @escaping (String) -> Void,
        onToolCall: @escaping (String, String) -> Void
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
                    onToolCall: onToolCall
                )

                let endTime = Date()
                let totalTime = endTime.timeIntervalSince(startTime) * 1000
                let timeToFirstToken = (firstTokenTime ?? endTime).timeIntervalSince(startTime) * 1000
                let tokensPerSecond = totalTime > 0 ? Double(tokenCount) / (totalTime / 1000) : 0

                self.lastStats = GenerationStats(
                    tokenCount: Double(tokenCount),
                    tokensPerSecond: tokensPerSecond,
                    timeToFirstToken: timeToFirstToken,
                    totalTime: totalTime
                )

                log("Stream with tools complete - \(tokenCount) tokens")
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
                let argsJson = self.dictionaryToJson(argsDict)

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
                    let resultJson = self.dictionaryToJson(resultDict)

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

    private func dictionaryToJson(_ dict: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return json
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
