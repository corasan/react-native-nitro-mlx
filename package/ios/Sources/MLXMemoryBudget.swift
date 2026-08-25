internal import MLX

/// Unbounded, MLX's buffer cache grows toward Metal's
/// recommendedMaxWorkingSetSize during sustained inference and walks the
/// process into iOS's Jetsam limit (observed: 2.4 GB footprint, OOM kill,
/// no crash log). 20 MB is the mlx-swift running-on-ios recommendation.
///
/// One shared knob for every modality (LLM, embeddings, STT, TTS): the limit
/// is process-global, so whichever module loads last wins.
enum MLXMemoryBudget {
    static let recommendedCacheLimit = 20 * 1024 * 1024

    static func applyRecommendedCacheLimit() {
        Memory.cacheLimit = recommendedCacheLimit
    }
}
