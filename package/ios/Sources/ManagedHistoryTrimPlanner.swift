import Foundation

struct ManagedHistoryTrimPlan {
    let removalCount: Int
    let tokenCount: Int
    let fitsBudget: Bool
}

enum ManagedHistoryTrimPlanner {
    static func plan(
        initialTokenCount: Int,
        maxContextTokens: Int,
        maxRemovableMessages: Int,
        tokenCountAfterRemoving: (Int) async throws -> Int
    ) async throws -> ManagedHistoryTrimPlan? {
        guard initialTokenCount > maxContextTokens else { return nil }
        guard maxRemovableMessages > 0 else { return nil }

        var tokenCountCache: [Int: Int] = [0: initialTokenCount]

        func tokenCount(afterRemoving removalCount: Int) async throws -> Int {
            if let cached = tokenCountCache[removalCount] {
                return cached
            }

            let count = try await tokenCountAfterRemoving(removalCount)
            tokenCountCache[removalCount] = count
            return count
        }

        var lowerBound = 1
        var upperBound = maxRemovableMessages
        var fittingRemovalCount: Int?
        var fittingTokenCount: Int?

        while lowerBound <= upperBound {
            let removalCount = lowerBound + (upperBound - lowerBound) / 2
            let count = try await tokenCount(afterRemoving: removalCount)

            if count <= maxContextTokens {
                fittingRemovalCount = removalCount
                fittingTokenCount = count
                upperBound = removalCount - 1
            } else {
                lowerBound = removalCount + 1
            }
        }

        if let fittingRemovalCount, let fittingTokenCount {
            return ManagedHistoryTrimPlan(
                removalCount: fittingRemovalCount,
                tokenCount: fittingTokenCount,
                fitsBudget: true
            )
        }

        let finalTokenCount = try await tokenCount(afterRemoving: maxRemovableMessages)
        return ManagedHistoryTrimPlan(
            removalCount: maxRemovableMessages,
            tokenCount: finalTokenCount,
            fitsBudget: false
        )
    }
}
