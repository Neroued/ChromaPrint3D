import type { ConvertAnyParams, DefaultConfig } from '../../types'

type InitialConvertParamsArgs = {
  defaults: DefaultConfig
  targetWidthMm: number
  targetHeightMm: number
  pixelMm: number
  dbNames: string[]
}

export function createInitialConvertParams({
  defaults,
  targetWidthMm,
  targetHeightMm,
  pixelMm,
  dbNames,
}: InitialConvertParamsArgs): ConvertAnyParams {
  return {
    print_mode: defaults.print_mode,
    color_space: defaults.color_space,
    max_width: 0,
    max_height: 0,
    target_width_mm: targetWidthMm,
    target_height_mm: targetHeightMm,
    scale: defaults.scale,
    k_candidates: defaults.k_candidates,
    cluster_method: 'kmeans',
    cluster_count: defaults.cluster_count,
    dither: 'none',
    dither_strength: defaults.dither_strength,
    model_enable: defaults.model_enable,
    model_only: defaults.model_only,
    model_threshold: defaults.model_threshold,
    model_margin: defaults.model_margin,
    flip_y: defaults.flip_y,
    pixel_mm: pixelMm,
    layer_height_mm: defaults.layer_height_mm,
    base_layers: typeof defaults.base_layers === 'number' ? defaults.base_layers : undefined,
    double_sided: defaults.double_sided ?? false,
    generate_preview: defaults.generate_preview,
    generate_source_mask: defaults.generate_source_mask,
    db_names: dbNames,
    tessellation_tolerance_mm: 0.03,
  }
}
