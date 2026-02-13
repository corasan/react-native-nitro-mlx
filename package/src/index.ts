export {
  LLM,
  type EventCallback,
  type Message,
  type ToolCallInfo,
  type ToolCallUpdate,
} from './llm'
export { ModelManager } from './modelManager'
export {
  MLXModel,
  MLXModels,
  ModelFamily,
  type ModelInfo,
  ModelProvider,
  type ModelQuantization,
  type ModelType,
} from './models'
export type {
  GenerationStats,
  LLM as LLMSpec,
  LLMLoadOptions,
  StreamEvent,
  GenerationStartEvent,
  TokenEvent,
  ThinkingStartEvent,
  ThinkingChunkEvent,
  ThinkingEndEvent,
  ToolCallStartEvent,
  ToolCallExecutingEvent,
  ToolCallCompletedEvent,
  ToolCallFailedEvent,
  GenerationEndEvent,
  ToolDefinition,
  ToolParameter,
  ToolParameterType,
} from './specs/LLM.nitro'
export type { ModelManager as ModelManagerSpec } from './specs/ModelManager.nitro'
export { createTool, type TypeSafeToolDefinition } from './tool-utils'
export { TTS } from './tts'
export type {
  TTS as TTSSpec,
  TTSLoadOptions,
  TTSGenerateOptions,
} from './specs/TTS.nitro'
export { STT } from './stt'
export type {
  STT as STTSpec,
  STTLoadOptions,
  STTTranscriptionInfo,
} from './specs/STT.nitro'
