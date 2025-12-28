import Foundation
import NitroModules
internal import MLX
internal import MLXLLM
internal import MLXLMCommon

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

    func load(modelId: String, options: LLMLoadOptions?) throws -> Promise<Void> {
        self.loadTask?.cancel()

        return Promise.async { [self] in
            let task = Task { @MainActor in
                MLX.GPU.set(cacheLimit: 2000000)

                self.currentTask?.cancel()
                self.currentTask = nil
                self.session = nil
                self.container = nil
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
