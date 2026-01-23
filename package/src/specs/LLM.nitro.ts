import type { AnyMap, HybridObject } from 'react-native-nitro-modules'

/**
 * Statistics from the last text generation.
 */
export interface GenerationStats {
  tokenCount: number
  tokensPerSecond: number
  timeToFirstToken: number
  totalTime: number
}

export interface GenerationStartEvent {
  type: 'generation_start'
  timestamp: number
}

export interface TokenEvent {
  type: 'token'
  token: string
}

export interface ThinkingStartEvent {
  type: 'thinking_start'
  timestamp: number
}

export interface ThinkingChunkEvent {
  type: 'thinking_chunk'
  chunk: string
}

export interface ThinkingEndEvent {
  type: 'thinking_end'
  content: string
  timestamp: number
}

export interface ToolCallStartEvent {
  type: 'tool_call_start'
  id: string
  name: string
  arguments: string
}

export interface ToolCallExecutingEvent {
  type: 'tool_call_executing'
  id: string
}

export interface ToolCallCompletedEvent {
  type: 'tool_call_completed'
  id: string
  result: string
}

export interface ToolCallFailedEvent {
  type: 'tool_call_failed'
  id: string
  error: string
}

export interface GenerationEndEvent {
  type: 'generation_end'
  content: string
  stats: GenerationStats
}

export type StreamEvent =
  | GenerationStartEvent
  | TokenEvent
  | ThinkingStartEvent
  | ThinkingChunkEvent
  | ThinkingEndEvent
  | ToolCallStartEvent
  | ToolCallExecutingEvent
  | ToolCallCompletedEvent
  | ToolCallFailedEvent
  | GenerationEndEvent

export interface LLMMessage {
  role: string
  content: string
}

/**
 * Parameter definition for a tool.
 */
export interface ToolParameter {
  name: string
  type: string
  description: string
  required: boolean
}

/**
 * Tool definition that can be called by the model.
 */
export interface ToolDefinition {
  name: string
  description: string
  parameters: ToolParameter[]
  handler: (args: AnyMap) => Promise<AnyMap>
}

/** Options for loading a model.
 */
export interface LLMLoadOptions {
  /** Callback invoked with loading progress (0-1) */
  onProgress?: (progress: number) => void
  /** Additional context to provide to the model */
  additionalContext?: LLMMessage[]
  /** Whether to automatically manage message history */
  manageHistory?: boolean
  /** Tools available for the model to call */
  tools?: ToolDefinition[]
}

/**
 * Low-level LLM interface for text generation using MLX.
 * @internal Use the `LLM` export from `react-native-nitro-mlx` instead.
 */
export interface LLM extends HybridObject<{ ios: 'swift' }> {
  /**
   * Load a model into memory. Downloads from HuggingFace if not already cached.
   * @param modelId - HuggingFace model ID (e.g., 'mlx-community/Qwen3-0.6B-4bit')
   * @param options - Callback invoked with loading progress (0-1)
   */
  load(modelId: string, options?: LLMLoadOptions): Promise<void>

  /**
   * Generate a complete response for a prompt.
   * @param prompt - The input text to generate a response for
   * @returns The generated text
   */
  generate(prompt: string): Promise<string>

  /**
   * Stream a response token by token with optional tool calling support.
   * Tools are automatically executed when the model calls them.
   * @param prompt - The input text to generate a response for
   * @param onToken - Callback invoked for each generated token
   * @param onToolCall - Optional callback invoked when a tool is called (for UI feedback)
   * @returns The complete generated text
   */
  stream(
    prompt: string,
    onToken: (token: string) => void,
    onToolCall?: (toolName: string, args: string) => void,
  ): Promise<string>

  streamWithEvents(
    prompt: string,
    onEvent: (eventJson: string) => void,
  ): Promise<string>

  /**
   * Stop the current generation.
   */
  stop(): void

  /**
   * Unload the current model and release memory.
   */
  unload(): void

  /**
   * Get statistics from the last generation.
   * @returns Statistics including token count, speed, and timing
   */
  getLastGenerationStats(): GenerationStats

  /**
   * Get the message history if management is enabled.
   * @returns Array of messages in the history
   */
  getHistory(): LLMMessage[]

  /**
   * Clear the message history.
   */
  clearHistory(): void

  /** Whether a model is currently loaded */
  readonly isLoaded: boolean
  /** Whether text is currently being generated */
  readonly isGenerating: boolean
  /** The ID of the currently loaded model */
  readonly modelId: string

  /** Enable debug logging */
  debug: boolean
  /** System prompt used when loading the model */
  systemPrompt: string
}

/**
 * Supported parameter types for tool definitions.
 * Used for type safety in createTool().
 */
export type ToolParameterType = 'string' | 'number' | 'boolean' | 'array' | 'object'
