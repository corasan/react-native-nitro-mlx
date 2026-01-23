import Foundation
import NitroModules

class StreamEventEmitter {
    private let callback: (String) -> Void

    init(callback: @escaping (String) -> Void) {
        self.callback = callback
    }

    private func emit(_ dict: [String: Any]) {
        callback(dictionaryToJson(dict))
    }

    private func timestamp() -> Double {
        Date().timeIntervalSince1970 * 1000
    }

    func emitGenerationStart() {
        emit([
            "type": "generation_start",
            "timestamp": timestamp()
        ])
    }

    func emitToken(_ token: String) {
        emit([
            "type": "token",
            "token": token
        ])
    }

    func emitThinkingStart() {
        emit([
            "type": "thinking_start",
            "timestamp": timestamp()
        ])
    }

    func emitThinkingChunk(_ chunk: String) {
        emit([
            "type": "thinking_chunk",
            "chunk": chunk
        ])
    }

    func emitThinkingEnd(_ content: String) {
        emit([
            "type": "thinking_end",
            "content": content,
            "timestamp": timestamp()
        ])
    }

    func emitToolCallStart(id: String, name: String, arguments: String) {
        emit([
            "type": "tool_call_start",
            "id": id,
            "name": name,
            "arguments": arguments
        ])
    }

    func emitToolCallExecuting(id: String) {
        emit([
            "type": "tool_call_executing",
            "id": id
        ])
    }

    func emitToolCallCompleted(id: String, result: String) {
        emit([
            "type": "tool_call_completed",
            "id": id,
            "result": result
        ])
    }

    func emitToolCallFailed(id: String, error: String) {
        emit([
            "type": "tool_call_failed",
            "id": id,
            "error": error
        ])
    }

    func emitGenerationEnd(content: String, stats: GenerationStats) {
        emit([
            "type": "generation_end",
            "content": content,
            "stats": [
                "tokenCount": stats.tokenCount,
                "tokensPerSecond": stats.tokensPerSecond,
                "timeToFirstToken": stats.timeToFirstToken,
                "totalTime": stats.totalTime
            ]
        ])
    }
}
