import Foundation

enum TestFailure: Error, CustomStringConvertible {
    case failed(String)

    var description: String {
        switch self {
        case .failed(let message):
            return message
        }
    }
}

func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() {
        throw TestFailure.failed(message)
    }
}

@main
struct ManagedHistoryTrimPlannerSpyTests {
    static func main() async throws {
        try await findsSmallestFittingRemovalWithLogarithmicTokenProbes()
        try await trimsToMaxRemovableWhenBudgetStillCannotFit()
        try await skipsWorkWhenInitialPromptAlreadyFits()
        print("ManagedHistoryTrimPlannerSpyTests passed")
    }

    private static func findsSmallestFittingRemovalWithLogarithmicTokenProbes() async throws {
        var probedRemovalCounts: [Int] = []

        let plan = try await ManagedHistoryTrimPlanner.plan(
            initialTokenCount: 220,
            maxContextTokens: 100,
            maxRemovableMessages: 16,
            tokenCountAfterRemoving: { removalCount in
                probedRemovalCounts.append(removalCount)
                return 220 - removalCount * 10
            }
        )

        try expect(plan?.removalCount == 12, "expected to remove the smallest fitting prefix")
        try expect(plan?.tokenCount == 100, "expected final token count at the budget")
        try expect(plan?.fitsBudget == true, "expected plan to fit the token budget")
        try expect(probedRemovalCounts.count <= 5, "expected logarithmic probe count")
        try expect(
            Set(probedRemovalCounts).count == probedRemovalCounts.count,
            "expected token-count cache to avoid duplicate probes"
        )
    }

    private static func trimsToMaxRemovableWhenBudgetStillCannotFit() async throws {
        var probedRemovalCounts: [Int] = []

        let plan = try await ManagedHistoryTrimPlanner.plan(
            initialTokenCount: 500,
            maxContextTokens: 100,
            maxRemovableMessages: 4,
            tokenCountAfterRemoving: { removalCount in
                probedRemovalCounts.append(removalCount)
                return 500 - removalCount * 20
            }
        )

        try expect(plan?.removalCount == 4, "expected to preserve pinned/recent messages")
        try expect(plan?.tokenCount == 420, "expected final count after max removal")
        try expect(plan?.fitsBudget == false, "expected budget to remain exceeded")
        try expect(
            probedRemovalCounts.last == 4,
            "expected final max-removal count to be measured for warning state"
        )
    }

    private static func skipsWorkWhenInitialPromptAlreadyFits() async throws {
        var prepareCalls = 0

        let plan = try await ManagedHistoryTrimPlanner.plan(
            initialTokenCount: 80,
            maxContextTokens: 100,
            maxRemovableMessages: 16,
            tokenCountAfterRemoving: { _ in
                prepareCalls += 1
                return 0
            }
        )

        try expect(plan == nil, "expected no trim plan when prompt already fits")
        try expect(prepareCalls == 0, "expected no extra tokenization when already in budget")
    }
}
