import Foundation

class ThinkingStateMachine {
    enum Output {
        case token(String)
        case thinkingStart
        case thinkingChunk(String)
        case thinkingEnd(String)
    }

    private enum State {
        case idle
        case bufferingOpenTag(String)
        case inThinking
        case bufferingCloseTag(String)
    }

    private var state: State = .idle
    private var thinkingContent: String = ""

    private let openTag = "<think>"
    private let closeTag = "</think>"

    func process(token: String) -> [Output] {
        var outputs: [Output] = []
        var remaining = token

        while !remaining.isEmpty {
            switch state {
            case .idle:
                outputs.append(contentsOf: processIdle(&remaining))

            case .bufferingOpenTag(let buffer):
                outputs.append(contentsOf: processBufferingOpenTag(buffer: buffer, remaining: &remaining))

            case .inThinking:
                outputs.append(contentsOf: processInThinking(&remaining))

            case .bufferingCloseTag(let buffer):
                outputs.append(contentsOf: processBufferingCloseTag(buffer: buffer, remaining: &remaining))
            }
        }

        return outputs
    }

    func flush() -> [Output] {
        var outputs: [Output] = []

        switch state {
        case .bufferingOpenTag(let buffer):
            if !buffer.isEmpty {
                outputs.append(.token(buffer))
            }
        case .bufferingCloseTag(let buffer):
            if !buffer.isEmpty {
                outputs.append(.thinkingChunk(buffer))
            }
            outputs.append(.thinkingEnd(thinkingContent + buffer))
        case .inThinking:
            outputs.append(.thinkingEnd(thinkingContent))
        case .idle:
            break
        }

        state = .idle
        thinkingContent = ""
        return outputs
    }

    private func processIdle(_ remaining: inout String) -> [Output] {
        var outputs: [Output] = []

        if let tagRange = remaining.range(of: openTag) {
            let before = String(remaining[..<tagRange.lowerBound])
            if !before.isEmpty {
                outputs.append(.token(before))
            }
            outputs.append(.thinkingStart)
            state = .inThinking
            thinkingContent = ""
            remaining = String(remaining[tagRange.upperBound...])
        } else if let partialMatch = findPartialMatch(remaining, target: openTag) {
            let before = String(remaining[..<partialMatch.range.lowerBound])
            if !before.isEmpty {
                outputs.append(.token(before))
            }
            state = .bufferingOpenTag(partialMatch.matched)
            remaining = ""
        } else {
            outputs.append(.token(remaining))
            remaining = ""
        }

        return outputs
    }

    private func processBufferingOpenTag(buffer: String, remaining: inout String) -> [Output] {
        var outputs: [Output] = []
        let combined = buffer + remaining

        if let tagRange = combined.range(of: openTag) {
            let before = String(combined[..<tagRange.lowerBound])
            if !before.isEmpty {
                outputs.append(.token(before))
            }
            outputs.append(.thinkingStart)
            state = .inThinking
            thinkingContent = ""
            remaining = String(combined[tagRange.upperBound...])
        } else if openTag.hasPrefix(combined) {
            state = .bufferingOpenTag(combined)
            remaining = ""
        } else if let partialMatch = findPartialMatch(combined, target: openTag) {
            let before = String(combined[..<partialMatch.range.lowerBound])
            if !before.isEmpty {
                outputs.append(.token(before))
            }
            state = .bufferingOpenTag(partialMatch.matched)
            remaining = ""
        } else {
            outputs.append(.token(combined))
            state = .idle
            remaining = ""
        }

        return outputs
    }

    private func processInThinking(_ remaining: inout String) -> [Output] {
        var outputs: [Output] = []

        if let tagRange = remaining.range(of: closeTag) {
            let before = String(remaining[..<tagRange.lowerBound])
            if !before.isEmpty {
                thinkingContent += before
                outputs.append(.thinkingChunk(before))
            }
            outputs.append(.thinkingEnd(thinkingContent))
            state = .idle
            thinkingContent = ""
            remaining = String(remaining[tagRange.upperBound...])
        } else if let partialMatch = findPartialMatch(remaining, target: closeTag) {
            let before = String(remaining[..<partialMatch.range.lowerBound])
            if !before.isEmpty {
                thinkingContent += before
                outputs.append(.thinkingChunk(before))
            }
            state = .bufferingCloseTag(partialMatch.matched)
            remaining = ""
        } else {
            thinkingContent += remaining
            outputs.append(.thinkingChunk(remaining))
            remaining = ""
        }

        return outputs
    }

    private func processBufferingCloseTag(buffer: String, remaining: inout String) -> [Output] {
        var outputs: [Output] = []
        let combined = buffer + remaining

        if let tagRange = combined.range(of: closeTag) {
            let before = String(combined[..<tagRange.lowerBound])
            if !before.isEmpty {
                thinkingContent += before
                outputs.append(.thinkingChunk(before))
            }
            outputs.append(.thinkingEnd(thinkingContent))
            state = .idle
            thinkingContent = ""
            remaining = String(combined[tagRange.upperBound...])
        } else if closeTag.hasPrefix(combined) {
            state = .bufferingCloseTag(combined)
            remaining = ""
        } else if let partialMatch = findPartialMatch(combined, target: closeTag) {
            let before = String(combined[..<partialMatch.range.lowerBound])
            if !before.isEmpty {
                thinkingContent += before
                outputs.append(.thinkingChunk(before))
            }
            state = .bufferingCloseTag(partialMatch.matched)
            remaining = ""
        } else {
            thinkingContent += combined
            outputs.append(.thinkingChunk(combined))
            state = .inThinking
            remaining = ""
        }

        return outputs
    }

    private func findPartialMatch(_ str: String, target: String) -> (range: Range<String.Index>, matched: String)? {
        for i in 1..<target.count {
            let suffix = String(str.suffix(i))
            let prefix = String(target.prefix(i))
            if suffix == prefix {
                let startIndex = str.index(str.endIndex, offsetBy: -i)
                return (startIndex..<str.endIndex, suffix)
            }
        }
        return nil
    }
}
