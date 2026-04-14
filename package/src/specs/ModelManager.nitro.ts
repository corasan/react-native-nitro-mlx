import type { HybridObject } from 'react-native-nitro-modules'

/**
 * Low-level interface for managing MLX model downloads.
 * @internal Use the `ModelManager` export from `react-native-nitro-mlx` instead.
 */
export interface ModelManager extends HybridObject<{ ios: 'swift' }> {
  /**
   * Download a model from HuggingFace.
   * @param modelId - HuggingFace model ID (e.g., 'mlx-community/Qwen3-0.6B-4bit')
   * @param progressCallback - Callback invoked with download progress (0-1)
   * @returns Path to the downloaded model directory
   */
  download(modelId: string, progressCallback: (progress: number) => void): Promise<string>

  /**
   * Check if a model is already downloaded.
   * @param modelId - HuggingFace model ID
   * @returns True if the model is downloaded
   */
  isDownloaded(modelId: string): Promise<boolean>

  /**
   * Get a list of all downloaded model IDs.
   * @returns Array of downloaded model IDs
   */
  getDownloadedModels(): Promise<string[]>

  /**
   * Delete a downloaded model.
   * @param modelId - HuggingFace model ID
   */
  deleteModel(modelId: string): Promise<void>

  /**
   * Get the local filesystem path for a downloaded model.
   * @param modelId - HuggingFace model ID
   * @returns Path to the model directory
   */
  getModelPath(modelId: string): Promise<string>

  /**
   * Read the download manifest for a downloaded model.
   * Useful for verifying on-device that a fresh download completed and wrote
   * `.download-manifest.json` into the model directory.
   * @param modelId - HuggingFace model ID
   * @returns The raw manifest JSON contents
   */
  getDownloadManifest(modelId: string): Promise<string>

  /** Enable debug logging */
  debug: boolean
}
