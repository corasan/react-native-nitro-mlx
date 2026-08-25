import { describe, expect, it, mock, spyOn } from 'bun:test'
import type { LLMMessage, LLMTurnOutcome } from './turn'

mock.module('react-native-nitro-modules', () => ({
  NitroModules: {
    createHybridObject: () => {
      throw new Error('Native module should be mocked in this test')
    },
  },
}))

const { runToolLoop } = await import('./agent')
const { LLM } = await import('./llm')

const stats = {
  tokenCount: 1,
  tokensPerSecond: 1,
  timeToFirstToken: 1,
  totalTime: 1,
  toolExecutionTime: 0,
}
const usage = { promptTokens: 1, completionTokens: 1 }

function answerTurn(content: string): LLMTurnOutcome {
  return { finishReason: 'completed', content, toolCalls: [], usage, stats }
}

function toolCallTurn(calls: Array<{ id: string; name: string }>): LLMTurnOutcome {
  return {
    finishReason: 'tool_calls',
    content: '',
    toolCalls: calls.map(c => ({ ...c, arguments: {} })),
    usage,
    stats,
  }
}

describe('runToolLoop', () => {
  it('returns a plain answer after one step', async () => {
    const spy = spyOn(LLM, 'runTurn').mockResolvedValueOnce(answerTurn('42'))
    try {
      const result = await runToolLoop(
        [{ role: 'user', content: 'what is 6*7?' }],
        () => 'unused',
      )
      expect(result).toMatchObject({ steps: 1, stoppedAtMaxSteps: false })
      expect(result.outcome.content).toBe('42')
    } finally {
      spy.mockRestore()
    }
  })

  it('executes tool calls and feeds results back as tool messages', async () => {
    const requests: LLMMessage[][] = []
    const spy = spyOn(LLM, 'runTurn')
      .mockImplementationOnce(async request => {
        requests.push(request.messages)
        return toolCallTurn([
          { id: 'a', name: 'get_time' },
          { id: 'b', name: 'explode' },
          { id: 'c', name: 'not_registered' },
        ])
      })
      .mockImplementationOnce(async request => {
        requests.push(request.messages)
        return answerTurn('done')
      })
    try {
      const result = await runToolLoop(
        [{ role: 'user', content: 'go' }],
        {
          get_time: () => '12:00',
          explode: () => {
            throw new Error('boom')
          },
        },
        { contextId: 'ctx-1' },
      )

      expect(result).toMatchObject({ steps: 2, stoppedAtMaxSteps: false })
      expect(result.outcome.content).toBe('done')
      expect(requests[1]).toEqual([
        {
          role: 'tool',
          toolCallId: 'a',
          name: 'get_time',
          content: '12:00',
          isError: undefined,
        },
        {
          role: 'tool',
          toolCallId: 'b',
          name: 'explode',
          content: 'boom',
          isError: true,
        },
        {
          role: 'tool',
          toolCallId: 'c',
          name: 'not_registered',
          content: 'Unknown tool: not_registered',
          isError: true,
        },
      ])
    } finally {
      spy.mockRestore()
    }
  })

  it('stops at maxSteps when the model keeps calling tools', async () => {
    const spy = spyOn(LLM, 'runTurn').mockImplementation(async () =>
      toolCallTurn([{ id: 'x', name: 'loop' }]),
    )
    try {
      const result = await runToolLoop([{ role: 'user', content: 'go' }], () => 'again', {
        maxSteps: 3,
      })
      expect(result).toMatchObject({ steps: 3, stoppedAtMaxSteps: true })
      expect(spy).toHaveBeenCalledTimes(3)
    } finally {
      spy.mockRestore()
    }
  })

  it('rejects a non-positive maxSteps', async () => {
    await expect(
      runToolLoop([{ role: 'user', content: 'go' }], () => '', { maxSteps: 0 }),
    ).rejects.toThrow(/maxSteps/)
  })
})
