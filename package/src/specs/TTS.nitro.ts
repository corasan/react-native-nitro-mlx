import type { HybridObject } from 'react-native-nitro-modules'

export interface TTSLoadOptions {
  onProgress?: (progress: number) => void
}

export interface TTSGenerateOptions {
  voice?: string
  speed?: number
}

export interface TTS extends HybridObject<{ ios: 'swift' }> {
  readonly isLoaded: boolean
  readonly isGenerating: boolean
  readonly modelId: string
  readonly sampleRate: number

  load(modelId: string, options?: TTSLoadOptions): Promise<void>
  generate(
    text: string,
    options?: TTSGenerateOptions
  ): Promise<ArrayBuffer>
  stream(
    text: string,
    onAudioChunk: (audio: ArrayBuffer) => void,
    options?: TTSGenerateOptions
  ): Promise<void>
  stop(): void
  unload(): void
}
