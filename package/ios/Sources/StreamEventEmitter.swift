import Foundation
import NitroModules

struct StreamEventEmitter {
    private let callback: (String) -> Void
    private let encoder = JSONEncoder()

    init(callback: @escaping (String) -> Void) {
        self.callback = callback
    }

    private func emit<T: Encodable>(_ event: T) {
        guard let data = try? encoder.encode(event),
              let json = String(data: data, encoding: .utf8) else { return }
        callback(json)
    }

    private func timestamp() -> Double {
        Date().timeIntervalSince1970 * 1000
    }

    struct GenerationStartEvent: Encodable {
        let type = "generation_start"
        let timestamp: Double
    }

    struct TokenEvent: Encodable {
        let type = "token"
        let token: String
    }

    struct ThinkingStartEvent: Encodable {
        let type = "thinking_start"
        let timestamp: Double
    }

    struct ThinkingChunkEvent: Encodable {
        let type = "thinking_chunk"
        let chunk: String
    }

    struct ThinkingEndEvent: Encodable {
        let type = "thinking_end"
        let content: String
        let timestamp: Double
    }

    struct ToolCallStartEvent: Encodable {
        let type = "tool_call_start"
        let id: String
        let name: String
        let arguments: String
    }

    struct ToolCallExecutingEvent: Encodable {
        let type = "tool_call_executing"
        let id: String
    }

    struct ToolCallCompletedEvent: Encodable {
        let type = "tool_call_completed"
        let id: String
        let result: String
    }

    struct ToolCallFailedEvent: Encodable {
        let type = "tool_call_failed"
        let id: String
        let error: String
    }

    struct StatsPayload: Encodable {
        let tokenCount: Double
        let tokensPerSecond: Double
        let timeToFirstToken: Double
        let totalTime: Double
    }

    struct GenerationEndEvent: Encodable {
        let type = "generation_end"
        let content: String
        let stats: StatsPayload
    }

    func emitGenerationStart() {
        emit(GenerationStartEvent(timestamp: timestamp()))
    }

    func emitToken(_ token: String) {
        emit(TokenEvent(token: token))
    }

    func emitThinkingStart() {
        emit(ThinkingStartEvent(timestamp: timestamp()))
    }

    func emitThinkingChunk(_ chunk: String) {
        emit(ThinkingChunkEvent(chunk: chunk))
    }

    func emitThinkingEnd(_ content: String) {
        emit(ThinkingEndEvent(content: content, timestamp: timestamp()))
    }

    func emitToolCallStart(id: String, name: String, arguments: String) {
        emit(ToolCallStartEvent(id: id, name: name, arguments: arguments))
    }

    func emitToolCallExecuting(id: String) {
        emit(ToolCallExecutingEvent(id: id))
    }

    func emitToolCallCompleted(id: String, result: String) {
        emit(ToolCallCompletedEvent(id: id, result: result))
    }

    func emitToolCallFailed(id: String, error: String) {
        emit(ToolCallFailedEvent(id: id, error: error))
    }

    func emitGenerationEnd(content: String, stats: GenerationStats) {
        emit(GenerationEndEvent(
            content: content,
            stats: StatsPayload(
                tokenCount: stats.tokenCount,
                tokensPerSecond: stats.tokensPerSecond,
                timeToFirstToken: stats.timeToFirstToken,
                totalTime: stats.totalTime
            )
        ))
    }
}
