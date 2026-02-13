import { NitroModules } from 'react-native-nitro-modules'
import type {
  STT as STTSpec,
  STTLoadOptions,
} from './specs/STT.nitro'

let instance: STTSpec | null = null

function getInstance(): STTSpec {
  if (!instance) {
    instance = NitroModules.createHybridObject<STTSpec>('STT')
  }
  return instance
}

export const STT = {
  load(modelId: string, options?: STTLoadOptions): Promise<void> {
    return getInstance().load(modelId, options)
  },

  transcribe(audio: ArrayBuffer): Promise<string> {
    return getInstance().transcribe(audio)
  },

  transcribeStream(
    audio: ArrayBuffer,
    onToken: (token: string) => void
  ): Promise<string> {
    return getInstance().transcribeStream(audio, onToken)
  },

  startListening(): Promise<void> {
    return getInstance().startListening()
  },

  transcribeBuffer(): Promise<string> {
    return getInstance().transcribeBuffer()
  },

  stopListening(): Promise<string> {
    return getInstance().stopListening()
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

  get isTranscribing(): boolean {
    return getInstance().isTranscribing
  },

  get isListening(): boolean {
    return getInstance().isListening
  },

  get modelId(): string {
    return getInstance().modelId
  },
}
