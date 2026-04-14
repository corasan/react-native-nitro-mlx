import { describe, expect, it } from 'bun:test'
import {
  assertArrayBuffer,
  assertBoolean,
  assertNonEmptyString,
  createSafeCallback,
  safeJsonParse,
  validateLLMLoadOptions,
  validateTTSGenerateOptions,
} from './runtime'

describe('runtime guards', () => {
  it('accepts valid primitive inputs', () => {
    expect(assertNonEmptyString('mlx-community/Qwen3-0.6B-4bit', 'modelId')).toBe(
      'mlx-community/Qwen3-0.6B-4bit',
    )
    expect(assertBoolean(true, 'debug')).toBe(true)
    expect(assertArrayBuffer(new ArrayBuffer(8), 'audio').byteLength).toBe(8)
  })

  it('rejects invalid primitive inputs', () => {
    expect(() => assertNonEmptyString('   ', 'modelId')).toThrow('must be a non-empty string')
    expect(() => assertBoolean('true', 'debug')).toThrow('must be a boolean')
    expect(() => assertArrayBuffer(new ArrayBuffer(0), 'audio')).toThrow('must not be empty')
  })

  it('wraps callbacks so user exceptions do not escape', () => {
    const originalConsoleError = console.error
    const errors: unknown[][] = []
    console.error = (...args: unknown[]) => {
      errors.push(args)
    }

    try {
      const callback = createSafeCallback('LLM.stream onToken', () => {
        throw new Error('boom')
      })

      expect(() => callback?.('token')).not.toThrow()
      expect(errors).toHaveLength(1)
      expect(String(errors[0]?.[0])).toContain('callback threw')
    } finally {
      console.error = originalConsoleError
    }
  })

  it('rejects duplicate tool names', () => {
    expect(() =>
      validateLLMLoadOptions({
        tools: [
          {
            name: 'weather',
            description: 'Weather tool',
            parameters: [],
            handler: async () => ({}),
          },
          {
            name: 'weather',
            description: 'Duplicate weather tool',
            parameters: [],
            handler: async () => ({}),
          },
        ],
      }),
    ).toThrow('tools must have unique names')
  })

  it('rejects invalid TTS generation options', () => {
    expect(() => validateTTSGenerateOptions({ speed: 0 })).toThrow(
      'must be a positive finite number',
    )
    expect(() => validateTTSGenerateOptions({ voice: '   ' })).toThrow(
      'must be a non-empty string',
    )
  })

  it('falls back on malformed JSON payloads', () => {
    expect(safeJsonParse('{"ok":true}', { ok: false })).toEqual({ ok: true })
    expect(safeJsonParse('{bad json', { ok: false })).toEqual({ ok: false })
  })
})
