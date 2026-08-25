import { describe, expect, it } from 'bun:test'
import { getModelInfo, MLXModel, MLXModels } from './models'

describe('model catalog', () => {
  it('has exactly one ModelInfo entry per MLXModel member', () => {
    const enumIds = Object.values(MLXModel)
    const infoIds = MLXModels.map(m => m.id)
    expect(new Set(infoIds).size).toBe(infoIds.length)
    expect(infoIds.toSorted()).toEqual(enumIds.toSorted())
  })

  it('looks up curated info by model id', () => {
    const info = getModelInfo(MLXModel.Qwen3_1_7B_4bit)
    expect(info?.displayName).toBe('Qwen 3 1.7B (4-bit)')
    expect(info?.downloadSize).toBeGreaterThan(0)
    expect(getModelInfo('not/a-model')).toBeUndefined()
  })
})
