import { describe, expect, it } from 'vitest'
import type { LayerPreviewsSummary } from '../../types'
import {
  getAvailableLayerCount,
  getLayerArtifactIndexFromTop,
  getLayerArtifactKeyFromTop,
  getTopLayerPosition,
} from './layerPreview'

function makeSummary(patch: Partial<LayerPreviewsSummary> = {}): LayerPreviewsSummary {
  return {
    enabled: true,
    layers: 4,
    width: 128,
    height: 96,
    layer_order: 'Top2Bottom',
    palette: [],
    artifacts: ['layer-preview-0', 'layer-preview-1', 'layer-preview-2', 'layer-preview-3'],
    ...patch,
  }
}

describe('layerPreview mapping', () => {
  it('maps top semantic positions for Top2Bottom', () => {
    const summary = makeSummary({ layer_order: 'Top2Bottom' })
    expect(getLayerArtifactIndexFromTop(summary, 1)).toBe(0)
    expect(getLayerArtifactIndexFromTop(summary, 2)).toBe(1)
    expect(getLayerArtifactIndexFromTop(summary, 4)).toBe(3)
  })

  it('maps top semantic positions for Bottom2Top', () => {
    const summary = makeSummary({ layer_order: 'Bottom2Top' })
    expect(getLayerArtifactIndexFromTop(summary, 1)).toBe(3)
    expect(getLayerArtifactIndexFromTop(summary, 2)).toBe(2)
    expect(getLayerArtifactIndexFromTop(summary, 4)).toBe(0)
  })

  it('clamps top positions and resolves artifact key', () => {
    const summary = makeSummary({ layer_order: 'Top2Bottom' })
    expect(getTopLayerPosition(summary, -3)).toBe(1)
    expect(getTopLayerPosition(summary, 99)).toBe(4)
    expect(getLayerArtifactKeyFromTop(summary, -1)).toBe('layer-preview-0')
    expect(getLayerArtifactKeyFromTop(summary, 999)).toBe('layer-preview-3')
  })

  it('returns zero or null for unavailable layer previews', () => {
    const disabled = makeSummary({ enabled: false })
    const empty = makeSummary({ layers: 0, artifacts: [] })
    expect(getAvailableLayerCount(disabled)).toBe(0)
    expect(getAvailableLayerCount(empty)).toBe(0)
    expect(getLayerArtifactKeyFromTop(disabled, 1)).toBeNull()
    expect(getLayerArtifactKeyFromTop(empty, 1)).toBeNull()
  })
})
