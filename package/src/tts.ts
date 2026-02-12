import { NitroModules } from 'react-native-nitro-modules'
import type {
  TTS as TTSSpec,
  TTSLoadOptions,
  TTSGenerateOptions,
} from './specs/TTS.nitro'

let instance: TTSSpec | null = null

function getInstance(): TTSSpec {
  if (!instance) {
    instance = NitroModules.createHybridObject<TTSSpec>('TTS')
  }
  return instance
}

export const TTS = {
  load(modelId: string, options?: TTSLoadOptions): Promise<void> {
    return getInstance().load(modelId, options)
  },

  generate(
    text: string,
    options?: TTSGenerateOptions
  ): Promise<ArrayBuffer> {
    return getInstance().generate(text, options)
  },

  stream(
    text: string,
    onAudioChunk: (audio: ArrayBuffer) => void,
    options?: TTSGenerateOptions
  ): Promise<void> {
    return getInstance().stream(text, onAudioChunk, options)
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
