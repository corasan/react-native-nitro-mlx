import { NitroModules } from 'react-native-nitro-modules'
import type { GenerationStats, LLMLoadOptions, LLM as LLMSpec } from './specs/LLM.nitro'

let instance: LLMSpec | null = null

export type Message = {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export type ToolCallInfo = {
  name: string
  arguments: Record<string, unknown>
}

function getInstance(): LLMSpec {
  if (!instance) {
    instance = NitroModules.createHybridObject<LLMSpec>('LLM')
  }
  return instance
}

/**
 * LLM text generation using MLX on Apple Silicon.
 *
 * @example
 * ```ts
 * import { LLM } from 'react-native-nitro-mlx'
 *
 * // Load a model
 * await LLM.load('mlx-community/Qwen3-0.6B-4bit', progress => {
 *   console.log(`Loading: ${(progress * 100).toFixed(0)}%`)
 * })
 *
 * // Stream a response
 * await LLM.stream('Hello!', token => {
 *   process.stdout.write(token)
 * })
 *
 * // Get generation stats
 * const stats = LLM.getLastGenerationStats()
 * console.log(`${stats.tokensPerSecond} tokens/sec`)
 * ```
 */
export const LLM = {
  /**
   * Load a model into memory. Downloads the model from HuggingFace if not already cached.
   * @param modelId - HuggingFace model ID (e.g., 'mlx-community/Qwen3-0.6B-4bit')
   * @param options - Callback invoked with loading progress (0-1)
   */
  load(modelId: string, options: LLMLoadOptions): Promise<void> {
    return getInstance().load(modelId, options)
  },

  /**
   * Generate a complete response for a prompt. Blocks until generation is complete.
   * For streaming responses, use `stream()` instead.
   * @param prompt - The input text to generate a response for
   * @returns The complete generated text
   */
  generate(prompt: string): Promise<string> {
    return getInstance().generate(prompt)
  },

  /**
   * Stream a response token by token.
   * @param prompt - The input text to generate a response for
   * @param onToken - Callback invoked for each generated token
   * @returns The complete generated text
   */
  stream(prompt: string, onToken: (token: string) => void): Promise<string> {
    return getInstance().stream(prompt, onToken)
  },

  /**
   * Stream a response with tool calling support.
   * Tools must be provided when loading the model via `load()` options.
   * Tools are automatically executed when the model calls them.
   * @param prompt - The input text to generate a response for
   * @param onToken - Callback invoked for each generated token
   * @param onToolCall - Optional callback invoked when a tool is called (for UI feedback)
   * @returns The complete generated text
   */
  streamWithTools(
    prompt: string,
    onToken: (token: string) => void,
    onToolCall?: (info: ToolCallInfo) => void,
  ): Promise<string> {
    return getInstance().streamWithTools(
      prompt,
      onToken,
      (name: string, argsJson: string) => {
        if (onToolCall) {
          try {
            const args = JSON.parse(argsJson) as Record<string, unknown>
            onToolCall({ name, arguments: args })
          } catch {
            onToolCall({ name, arguments: {} })
          }
        }
      },
    )
  },

  /**
   * Stop the current generation. Safe to call even if not generating.
   */
  stop(): void {
    getInstance().stop()
  },

  /**
   * Unload the current model and release memory.
   * Call this when you're done with the model to free up memory.
   */
  unload(): void {
    getInstance().unload()
  },

  /**
   * Get statistics from the last generation.
   * @returns Statistics including token count, tokens/sec, TTFT, and total time
   */
  getLastGenerationStats(): GenerationStats {
    return getInstance().getLastGenerationStats()
  },

  /**
   * Get the message history if management is enabled.
   * @returns Array of messages in the history
   */
  getHistory(): Message[] {
    return getInstance().getHistory() as Message[]
  },

  /**
   * Clear the message history.
   */
  clearHistory(): void {
    getInstance().clearHistory()
  },

  /** Whether a model is currently loaded and ready for generation */
  get isLoaded(): boolean {
    return getInstance().isLoaded
  },

  /** Whether text is currently being generated */
  get isGenerating(): boolean {
    return getInstance().isGenerating
  },

  /** The ID of the currently loaded model, or empty string if none */
  get modelId(): string {
    return getInstance().modelId
  },

  /** Enable debug logging to console */
  get debug(): boolean {
    return getInstance().debug
  },

  set debug(value: boolean) {
    getInstance().debug = value
  },

  /**
   * System prompt used when loading the model.
   * Set this before calling `load()`. Changes require reloading the model.
   * @default "You are a helpful assistant."
   */
  get systemPrompt(): string {
    return getInstance().systemPrompt
  },

  set systemPrompt(value: string) {
    getInstance().systemPrompt = value
  },
}
