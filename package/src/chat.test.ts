import { describe, expect, it, mock, spyOn } from 'bun:test'
import type { GenerationStats } from './specs/LLM.nitro'

mock.module('react-native-nitro-modules', () => {
  return {
    NitroModules: {
      createHybridObject: () => {
        throw new Error('Native module should be mocked in this test')
      },
    },
  }
})

const { ChatSession } = await import('./chat')
const { LLM } = await import('./llm')

describe('ChatSession terminal outcomes', () => {
  it('resolves a failed started turn with partial content and stats', async () => {
    const stats: GenerationStats = {
      tokenCount: 3,
      tokensPerSecond: 6,
      timeToFirstToken: 120,
      totalTime: 500,
      toolExecutionTime: 0,
    }

    const loadSpy = spyOn(LLM, 'load').mockResolvedValue(undefined)
    const streamSpy = spyOn(LLM, 'streamWithEvents').mockImplementation(
      async (_prompt, onEvent) => {
        const outcome = {
          content: 'partial',
          stats,
          finishReason: 'failed' as const,
          error: 'Generation failed during generate: boom',
          stage: 'generate',
        }
        onEvent({ type: 'token', token: 'partial' })
        onEvent({
          type: 'generation_outcome',
          outcome,
        })
        return outcome
      },
    )

    try {
      const chat = new ChatSession({ modelId: 'test/model' })
      await chat.load()

      const message = await chat.sendMessage('hello')
      expect(chat.state.lastStats).toEqual(stats)
      expect(chat.state.status).toBe('error')
      expect(message).toMatchObject({
        role: 'assistant',
        content: 'partial',
        stats,
        error: 'Generation failed during generate: boom',
        outcome: { finishReason: 'failed' },
      })
    } finally {
      loadSpy.mockRestore()
      streamSpy.mockRestore()
    }
  })

  it('commits a stopped turn as a successful partial assistant message', async () => {
    const stats: GenerationStats = {
      tokenCount: 2,
      tokensPerSecond: 4,
      timeToFirstToken: 50,
      totalTime: 250,
      toolExecutionTime: 0,
    }
    const loadSpy = spyOn(LLM, 'load').mockResolvedValue(undefined)
    const streamSpy = spyOn(LLM, 'streamWithEvents').mockImplementation(
      async (_prompt, onEvent) => {
        const outcome = {
          content: 'partial',
          thinking: 'reasoning',
          stats,
          finishReason: 'stopped' as const,
        }
        onEvent({ type: 'token', token: 'partial' })
        onEvent({ type: 'generation_outcome', outcome })
        return outcome
      },
    )

    try {
      const chat = new ChatSession({
        modelId: 'test/model',
        onToken: () => {
          throw new Error('observer failure')
        },
        onMessage: () => {
          throw new Error('observer failure')
        },
      })
      await chat.load()

      await expect(chat.sendMessage('hello')).resolves.toMatchObject({
        content: 'partial',
        thinking: 'reasoning',
        outcome: { finishReason: 'stopped' },
      })
      expect(chat.state.status).toBe('done')
      expect(chat.state.isGenerating).toBe(false)
    } finally {
      loadSpy.mockRestore()
      streamSpy.mockRestore()
    }
  })
})

describe('ChatSession construction and load guards', () => {
  it('rejects a blank systemPrompt in the constructor instead of wedging load()', () => {
    expect(() => new ChatSession({ modelId: 'test/model', systemPrompt: '   ' })).toThrow(
      /systemPrompt/,
    )
  })

  it('routes a load failure through onError and leaves status error', async () => {
    const loadSpy = spyOn(LLM, 'load').mockRejectedValue(new Error('no disk'))
    try {
      let seenError: Error | null = null
      const chat = new ChatSession({
        modelId: 'test/model',
        onError: err => {
          seenError = err
        },
      })
      await expect(chat.load()).rejects.toThrow('no disk')
      expect(chat.state.status).toBe('error')
      expect(String(seenError)).toContain('no disk')
    } finally {
      loadSpy.mockRestore()
    }
  })
})

describe('ChatSession thinking preservation', () => {
  it('keeps accumulated thinking when the outcome omits it', async () => {
    const stats: GenerationStats = {
      tokenCount: 2,
      tokensPerSecond: 4,
      timeToFirstToken: 50,
      totalTime: 250,
      toolExecutionTime: 0,
    }
    const loadSpy = spyOn(LLM, 'load').mockResolvedValue(undefined)
    const streamSpy = spyOn(LLM, 'streamWithEvents').mockImplementation(
      async (_prompt, onEvent) => {
        // A stop racing the thinking accumulator: thinking streamed, but the
        // terminal outcome carries no `thinking` field.
        const outcome = {
          content: 'partial',
          stats,
          finishReason: 'stopped' as const,
        }
        onEvent({ type: 'thinking_start', timestamp: 1 })
        onEvent({ type: 'thinking_chunk', chunk: 'because' })
        onEvent({ type: 'thinking_end', content: 'because', timestamp: 2 })
        onEvent({ type: 'token', token: 'partial' })
        onEvent({ type: 'generation_outcome', outcome })
        return outcome
      },
    )

    try {
      const chat = new ChatSession({ modelId: 'test/model' })
      await chat.load()
      const message = await chat.sendMessage('hello')
      expect(message.thinking).toBe('because')
    } finally {
      loadSpy.mockRestore()
      streamSpy.mockRestore()
    }
  })
})

describe('ChatSession abort before generation', () => {
  it('rolls back the exchange and throws AbortError when a callback aborts', async () => {
    const loadSpy = spyOn(LLM, 'load').mockResolvedValue(undefined)
    const streamSpy = spyOn(LLM, 'streamWithEvents').mockImplementation(async () => {
      throw new Error('generation must not start after a pre-start abort')
    })
    try {
      const controller = new AbortController()
      const chat = new ChatSession({
        modelId: 'test/model',
        onMessage: () => controller.abort(),
      })
      await chat.load()

      let caught: unknown
      try {
        await chat.sendMessage('hello', { signal: controller.signal })
      } catch (error) {
        caught = error
      }
      expect(caught).toMatchObject({ name: 'AbortError' })
      expect(streamSpy).not.toHaveBeenCalled()
      expect(chat.messages).toHaveLength(0)
      expect(chat.state.status).toBe('idle')
    } finally {
      loadSpy.mockRestore()
      streamSpy.mockRestore()
    }
  })
})
