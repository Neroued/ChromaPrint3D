import type { ConvertAnyParams, ConvertRasterParams, ConvertVectorParams } from '../../types'

const RASTER_FIELDS: Array<keyof ConvertRasterParams> = [
  'db_names',
  'print_mode',
  'color_space',
  'max_width',
  'max_height',
  'target_width_mm',
  'target_height_mm',
  'scale',
  'k_candidates',
  'cluster_method',
  'cluster_count',
  'dither',
  'dither_strength',
  'allowed_channels',
  'model_enable',
  'model_only',
  'model_threshold',
  'model_margin',
  'flip_y',
  'pixel_mm',
  'layer_height_mm',
  'base_layers',
  'double_sided',
  'transparent_layer_mm',
  'generate_preview',
  'generate_source_mask',
  'nozzle_size',
  'face_orientation',
]

const VECTOR_FIELDS: Array<keyof ConvertVectorParams> = [
  'db_names',
  'print_mode',
  'color_space',
  'target_width_mm',
  'target_height_mm',
  'k_candidates',
  'model_enable',
  'model_only',
  'model_threshold',
  'model_margin',
  'flip_y',
  'layer_height_mm',
  'base_layers',
  'double_sided',
  'transparent_layer_mm',
  'allowed_channels',
  'generate_preview',
  'tessellation_tolerance_mm',
  'nozzle_size',
  'face_orientation',
]

function copyDefined<T extends object, K extends keyof T>(
  source: Partial<T>,
  fields: readonly K[],
): Partial<Pick<T, K>> {
  const out: Partial<Pick<T, K>> = {}
  for (const field of fields) {
    const value = source[field]
    if (value !== undefined) {
      out[field] = value as T[K]
    }
  }
  return out
}

export function buildRasterParams(params: ConvertAnyParams): ConvertRasterParams {
  const source = params as Partial<ConvertRasterParams>
  return copyDefined<ConvertRasterParams, keyof ConvertRasterParams>(source, RASTER_FIELDS)
}

export function buildVectorParams(params: ConvertAnyParams): ConvertVectorParams {
  const source = params as Partial<ConvertVectorParams>
  return copyDefined<ConvertVectorParams, keyof ConvertVectorParams>(source, VECTOR_FIELDS)
}
