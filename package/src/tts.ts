import { NitroModules } from 'react-native-nitro-modules'
import {
  assertNonEmptyString,
  createSafeCallback,
  validateTTSGenerateOptions,
  validateTTSLoadOptions,
} from './runtime'
import type {
  TTSGenerateOptions,
  TTSLoadOptions,
  TTS as TTSSpec,
} from './specs/TTS.nitro'

let instance: TTSSpec | null = null

function getInstance(): TTSSpec {
  if (!instance) {
    instance = NitroModules.createHybridObject<TTSSpec>('TTS')
  }
  if (!instance) {
    throw new Error('Failed to initialize the TTS Nitro module.')
  }
  return instance
}

export const TTS = {
  load(modelId: string, options?: TTSLoadOptions): Promise<void> {
    return getInstance().load(
      assertNonEmptyString(modelId, 'TTS modelId'),
      validateTTSLoadOptions(options),
    )
  },

  generate(text: string, options?: TTSGenerateOptions): Promise<ArrayBuffer> {
    return getInstance().generate(
      assertNonEmptyString(text, 'TTS text'),
      validateTTSGenerateOptions(options),
    )
  },

  stream(
    text: string,
    onAudioChunk: (audio: ArrayBuffer) => void,
    options?: TTSGenerateOptions,
  ): Promise<void> {
    return getInstance().stream(
      assertNonEmptyString(text, 'TTS text'),
      createSafeCallback('TTS.stream onAudioChunk', onAudioChunk) ?? (() => {}),
      validateTTSGenerateOptions(options),
    )
  },

  stop(): void {
    getInstance().stop()
  },

  unload(): void {
    getInstance().unload()
  },

  get isLoaded(): boolean {
    return getInstance().isLoaded
  },

  get isGenerating(): boolean {
    return getInstance().isGenerating
  },

  get modelId(): string {
    return getInstance().modelId
  },

  get sampleRate(): number {
    return getInstance().sampleRate
  },
}
