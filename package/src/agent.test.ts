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

describe('runToolLoop transcript and safety contracts', () => {
  it('accumulates the full transcript between cold-mode steps', async () => {
    const requests: LLMMessage[][] = []
    const spy = spyOn(LLM, 'runTurn')
      .mockImplementationOnce(async request => {
        requests.push(request.messages)
        return toolCallTurn([{ id: 'a', name: 'get_time' }])
      })
      .mockImplementationOnce(async request => {
        requests.push(request.messages)
        return answerTurn('done')
      })
    try {
      await runToolLoop([{ role: 'user', content: 'time?' }], {
        get_time: () => '12:00',
      })
      // Cold mode: the second request must carry the question, the assistant
      // tool-call turn, and the result — native cold turns are stateless.
      expect(requests[1]).toEqual([
        { role: 'user', content: 'time?' },
        {
          role: 'assistant',
          content: '',
          toolCalls: [{ id: 'a', name: 'get_time', arguments: {} }],
        },
        {
          role: 'tool',
          toolCallId: 'a',
          name: 'get_time',
          content: '12:00',
          isError: undefined,
        },
      ])
    } finally {
      spy.mockRestore()
    }
  })

  it('does not execute tools on the final maxSteps round', async () => {
    let executions = 0
    const spy = spyOn(LLM, 'runTurn').mockImplementation(async () =>
      toolCallTurn([{ id: 'x', name: 'loop' }]),
    )
    try {
      const result = await runToolLoop(
        [{ role: 'user', content: 'go' }],
        () => {
          executions += 1
          return 'r'
        },
        { maxSteps: 2 },
      )
      expect(result).toMatchObject({ steps: 2, stoppedAtMaxSteps: true })
      expect(result.outcome.toolCalls).toHaveLength(1)
      // The final round's calls are returned unexecuted, not fired-and-dropped.
      expect(executions).toBe(1)
    } finally {
      spy.mockRestore()
    }
  })

  it('converts a void executor return into an isError tool message', async () => {
    const requests: LLMMessage[][] = []
    const spy = spyOn(LLM, 'runTurn')
      .mockImplementationOnce(async () => toolCallTurn([{ id: 'v', name: 'fire' }]))
      .mockImplementationOnce(async request => {
        requests.push(request.messages)
        return answerTurn('ok')
      })
    try {
      await runToolLoop(
        [{ role: 'user', content: 'go' }],
        // SAFETY: deliberately returns undefined to exercise the guard.
        { fire: (() => undefined) as never },
        { contextId: 'ctx' },
      )
      expect(requests[0]?.[0]).toMatchObject({
        role: 'tool',
        toolCallId: 'v',
        isError: true,
      })
    } finally {
      spy.mockRestore()
    }
  })
})
