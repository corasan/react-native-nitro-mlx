internal import MLX

/// Unbounded, MLX's buffer cache grows toward Metal's
/// recommendedMaxWorkingSetSize during sustained inference and walks the
/// process into iOS's Jetsam limit (observed: 2.4 GB footprint, OOM kill,
/// no crash log). 20 MB is the mlx-swift running-on-ios recommendation.
///
/// One shared knob so every modality (LLM, embeddings, STT, TTS) applies the
/// same budget. The limit itself is process-global — whichever module loads
/// last wins — which is exactly why it must not be tuned per call site.
enum MLXMemoryBudget {
    static let recommendedCacheLimit = 20 * 1024 * 1024

    static func applyRecommendedCacheLimit() {
        Memory.cacheLimit = recommendedCacheLimit
    }
}
