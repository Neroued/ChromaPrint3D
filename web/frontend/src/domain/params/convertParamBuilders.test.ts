import { describe, expect, it } from 'vitest'
import { buildRasterParams, buildVectorParams } from './convertParamBuilders'
import type { ConvertAnyParams } from '../../types'

describe('convertParamBuilders', () => {
  const input: ConvertAnyParams = {
    db_names: ['db-a'],
    print_mode: '0.08x5',
    color_space: 'lab',
    max_width: 512,
    max_height: 512,
    target_width_mm: 120,
    target_height_mm: 100,
    scale: 1.2,
    k_candidates: 8,
    cluster_method: 'kmeans',
    cluster_count: 16,
    dither: 'blue_noise',
    dither_strength: 0.8,
    model_enable: true,
    model_only: false,
    model_threshold: 2.5,
    model_margin: 0.3,
    flip_y: true,
    pixel_mm: 0.42,
    layer_height_mm: 0.08,
    base_layers: 3,
    double_sided: true,
    generate_preview: true,
    generate_source_mask: true,
    nozzle_size: 'n02',
    face_orientation: 'facedown',
    tessellation_tolerance_mm: 0.15,
    gradient_dither: 'none',
    gradient_dither_strength: 0.5,
  }

  it('buildRasterParams should exclude vector-only fields', () => {
    const raster = buildRasterParams(input)
    expect(raster.max_width).toBe(512)
    expect(raster.cluster_method).toBe('kmeans')
    expect(raster.generate_source_mask).toBe(true)
    expect(raster.nozzle_size).toBe('n02')
    expect(raster.face_orientation).toBe('facedown')
    expect(raster.base_layers).toBe(3)
    expect(raster.double_sided).toBe(true)
    expect((raster as Record<string, unknown>).tessellation_tolerance_mm).toBeUndefined()
    expect((raster as Record<string, unknown>).gradient_dither).toBeUndefined()
  })

  it('buildVectorParams should exclude raster-only fields', () => {
    const vector = buildVectorParams(input)
    expect(vector.tessellation_tolerance_mm).toBe(0.15)
    expect(vector.gradient_dither).toBe('none')
    expect(vector.model_enable).toBe(true)
    expect(vector.model_only).toBe(false)
    expect(vector.model_threshold).toBe(2.5)
    expect(vector.model_margin).toBe(0.3)
    expect(vector.nozzle_size).toBe('n02')
    expect(vector.face_orientation).toBe('facedown')
    expect(vector.base_layers).toBe(3)
    expect(vector.double_sided).toBe(true)
    expect((vector as Record<string, unknown>).max_width).toBeUndefined()
    expect((vector as Record<string, unknown>).cluster_method).toBeUndefined()
    expect((vector as Record<string, unknown>).cluster_count).toBeUndefined()
    expect((vector as Record<string, unknown>).generate_source_mask).toBeUndefined()
  })
})
