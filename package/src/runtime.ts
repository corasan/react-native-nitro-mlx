import type { EmbeddingsLoadOptions } from './specs/Embeddings.nitro'
import type { LLMLoadOptions, ToolDefinition } from './specs/LLM.nitro'
import type { STTLoadOptions } from './specs/STT.nitro'
import type { TTSGenerateOptions, TTSLoadOptions } from './specs/TTS.nitro'

const ERROR_PREFIX = '[react-native-nitro-mlx]'
const runtimeConsole = (
  globalThis as { console?: { error?: (...args: unknown[]) => void } }
).console

function describeType(value: unknown): string {
  if (value === null) {
    return 'null'
  }
  if (value === undefined) {
    return 'undefined'
  }
  if (value instanceof ArrayBuffer) {
    return 'ArrayBuffer'
  }
  return typeof value
}

export function assertNonEmptyString(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be a non-empty string.`)
  }
  return value
}

export function assertArrayBuffer(value: unknown, name: string): ArrayBuffer {
  if (!(value instanceof ArrayBuffer)) {
    throw new TypeError(
      `${ERROR_PREFIX} ${name} must be an ArrayBuffer, received ${describeType(value)}.`,
    )
  }
  if (value.byteLength === 0) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must not be empty.`)
  }
  return value
}

export function assertBoolean(value: unknown, name: string): boolean {
  if (typeof value !== 'boolean') {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be a boolean.`)
  }
  return value
}

export function createSafeCallback<TArgs extends unknown[]>(
  name: string,
  callback?: ((...args: TArgs) => void) | null,
): ((...args: TArgs) => void) | undefined {
  if (callback == null) {
    return undefined
  }
  if (typeof callback !== 'function') {
    throw new TypeError(
      `${ERROR_PREFIX} ${name} must be a function, received ${describeType(callback)}.`,
    )
  }

  return (...args: TArgs) => {
    try {
      callback(...args)
    } catch (error) {
      runtimeConsole?.error?.(`${ERROR_PREFIX} ${name} callback threw.`, error)
    }
  }
}

function validateToolDefinitions(tools: ToolDefinition[]): ToolDefinition[] {
  const seenNames = new Set<string>()

  return tools.map((tool, index) => {
    const name = assertNonEmptyString(tool?.name, `tools[${index}].name`)
    if (seenNames.has(name)) {
      throw new TypeError(
        `${ERROR_PREFIX} tools must have unique names. Duplicate: '${name}'.`,
      )
    }
    seenNames.add(name)

    if (typeof tool.handler !== 'function') {
      throw new TypeError(`${ERROR_PREFIX} tools[${index}].handler must be a function.`)
    }

    return tool
  })
}

export function validateLLMLoadOptions(
  options?: LLMLoadOptions,
): LLMLoadOptions | undefined {
  if (!options) {
    return undefined
  }

  return {
    ...options,
    onProgress: createSafeCallback('LLM.load onProgress', options.onProgress),
    tools: options.tools ? validateToolDefinitions(options.tools) : options.tools,
  }
}

export function validateModelDownloadCallback(
  callback?: ((progress: number) => void) | null,
): ((progress: number) => void) | undefined {
  return createSafeCallback('ModelManager.download onProgress', callback)
}

export function validateSTTLoadOptions(
  options?: STTLoadOptions,
): STTLoadOptions | undefined {
  if (!options) {
    return undefined
  }

  return {
    ...options,
    onProgress: createSafeCallback('STT.load onProgress', options.onProgress),
  }
}

export function validateEmbeddingsLoadOptions(
  options?: EmbeddingsLoadOptions,
): EmbeddingsLoadOptions | undefined {
  if (!options) {
    return undefined
  }

  return {
    ...options,
    onProgress: createSafeCallback('Embeddings.load onProgress', options.onProgress),
  }
}

export function validateTTSLoadOptions(
  options?: TTSLoadOptions,
): TTSLoadOptions | undefined {
  if (!options) {
    return undefined
  }

  return {
    ...options,
    onProgress: createSafeCallback('TTS.load onProgress', options.onProgress),
  }
}

export function validateTTSGenerateOptions(
  options?: TTSGenerateOptions,
): TTSGenerateOptions | undefined {
  if (!options) {
    return undefined
  }

  if (options.voice !== undefined) {
    assertNonEmptyString(options.voice, 'TTS voice')
  }

  if (options.speed !== undefined) {
    if (!Number.isFinite(options.speed) || options.speed <= 0) {
      throw new RangeError(`${ERROR_PREFIX} TTS speed must be a positive finite number.`)
    }
  }

  return options
}

export function safeJsonParse<T>(value: string, fallback: T): T {
  try {
    return JSON.parse(value) as T
  } catch {
    return fallback
  }
}
