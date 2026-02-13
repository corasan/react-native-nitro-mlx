import type { HybridObject } from 'react-native-nitro-modules'

export interface STTLoadOptions {
  onProgress?: (progress: number) => void
}

export interface STTTranscriptionInfo {
  promptTokens: number
  generationTokens: number
  tokensPerSecond: number
  prefillTime: number
  generateTime: number
}

export interface STT extends HybridObject<{ ios: 'swift' }> {
  readonly isLoaded: boolean
  readonly isTranscribing: boolean
  readonly isListening: boolean
  readonly modelId: string

  load(modelId: string, options?: STTLoadOptions): Promise<void>

  transcribe(audio: ArrayBuffer): Promise<string>
  transcribeStream(
    audio: ArrayBuffer,
    onToken: (token: string) => void
  ): Promise<string>

  startListening(): Promise<void>
  transcribeBuffer(): Promise<string>
  stopListening(): Promise<string>

  stop(): void
  unload(): void
}
