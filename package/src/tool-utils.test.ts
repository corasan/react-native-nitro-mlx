import { describe, expect, it } from 'bun:test'
import { z } from 'zod'
import { createTool } from './tool-utils'

const timerTool = createTool({
  name: 'set_timer',
  description: 'Set a timer',
  arguments: z.object({
    minutes: z.number().describe('Duration in minutes'),
    label: z.string().optional().describe('Timer label'),
  }),
  handler: async args => ({ ok: true, minutes: args.minutes }),
})

describe('createTool', () => {
  it('converts a zod object schema to tool parameters', () => {
    expect(timerTool.name).toBe('set_timer')
    expect(timerTool.parameters).toEqual([
      {
        name: 'minutes',
        type: 'number',
        description: 'Duration in minutes',
        required: true,
      },
      { name: 'label', type: 'string', description: 'Timer label', required: false },
    ])
  })

  it('maps integer schemas to the number parameter type', () => {
    const tool = createTool({
      name: 'count',
      description: 'Count things',
      arguments: z.object({ n: z.number().int() }),
      handler: async () => ({}),
    })
    expect(tool.parameters[0]?.type).toBe('number')
  })

  it('runs the handler with parsed arguments', async () => {
    // SAFETY: the runtime hands the handler an AnyMap; a plain object matches.
    const result = await timerTool.handler({ minutes: 5 } as never)
    expect(result).toEqual({ ok: true, minutes: 5 })
  })

  it('throws a compact, model-recoverable error for invalid arguments', async () => {
    // SAFETY: deliberately mistyped input to exercise the validation path.
    const promise = timerTool.handler({ minutes: 'five' } as never)
    await expect(promise).rejects.toThrow(/Invalid arguments for tool "set_timer"/)
    await expect(promise).rejects.toThrow(/minutes/)
    await expect(promise).rejects.toThrow(/call the tool again/)
    // The raw ZodError JSON dump must not leak through.
    await expect(promise).rejects.not.toThrow(/"code"/)
  })
})
