import { roundTo } from '../../runtime/number'
import type { VectorizeParams } from '../../types'

export const defaultVectorizeParams = {
  num_colors: 16,
  min_region_area: 10,
  curve_fit_error: 0.8,
  corner_angle_threshold: 135,
  smoothing_spatial: 15,
  smoothing_color: 25,
  upscale_short_edge: 600,
  max_working_pixels: 3000000,
  slic_region_size: 20,
  edge_sensitivity: 0.8,
  refine_passes: 6,
  max_merge_color_dist: 150,
  thin_line_max_radius: 2.5,
  min_contour_area: 10,
  min_hole_area: 4.0,
  contour_simplify: 0.45,
  enable_coverage_fix: true,
  min_coverage_ratio: 0.998,
  svg_enable_stroke: true,
  svg_stroke_width: 0.5,
} satisfies Required<
  Pick<
    VectorizeParams,
    | 'num_colors'
    | 'min_region_area'
    | 'curve_fit_error'
    | 'corner_angle_threshold'
    | 'smoothing_spatial'
    | 'smoothing_color'
    | 'upscale_short_edge'
    | 'max_working_pixels'
    | 'slic_region_size'
    | 'edge_sensitivity'
    | 'refine_passes'
    | 'max_merge_color_dist'
    | 'thin_line_max_radius'
    | 'min_contour_area'
    | 'min_hole_area'
    | 'contour_simplify'
    | 'enable_coverage_fix'
    | 'min_coverage_ratio'
    | 'svg_enable_stroke'
    | 'svg_stroke_width'
  >
>

function normalizeNumber(
  value: unknown,
  fallback: number,
  min: number,
  max: number,
  integer = false,
  precision?: number,
): number {
  let out = typeof value === 'number' && Number.isFinite(value) ? value : fallback
  if (integer) out = Math.round(out)
  else if (typeof precision === 'number') out = roundTo(out, precision)
  return Math.min(max, Math.max(min, out))
}

function normalizeBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback
}

export function normalizeVectorizeParams(
  params: VectorizeParams,
  defaults: VectorizeParams = defaultVectorizeParams,
): VectorizeParams {
  return {
    num_colors: normalizeNumber(params.num_colors, defaults.num_colors ?? 16, 2, 256, true),
    curve_fit_error: normalizeNumber(
      params.curve_fit_error,
      defaults.curve_fit_error ?? 0.8,
      0.1,
      5,
      false,
      2,
    ),
    corner_angle_threshold: normalizeNumber(
      params.corner_angle_threshold,
      defaults.corner_angle_threshold ?? 135,
      90,
      175,
      false,
      1,
    ),
    smoothing_spatial: normalizeNumber(
      params.smoothing_spatial,
      defaults.smoothing_spatial ?? 15,
      0,
      50,
      false,
      1,
    ),
    smoothing_color: normalizeNumber(
      params.smoothing_color,
      defaults.smoothing_color ?? 25,
      0,
      80,
      false,
      1,
    ),
    upscale_short_edge: normalizeNumber(
      params.upscale_short_edge,
      defaults.upscale_short_edge ?? 600,
      0,
      2000,
      true,
    ),
    max_working_pixels: normalizeNumber(
      params.max_working_pixels,
      defaults.max_working_pixels ?? 3000000,
      0,
      100000000,
      true,
    ),
    slic_region_size: normalizeNumber(
      params.slic_region_size,
      defaults.slic_region_size ?? 20,
      0,
      100,
      true,
    ),
    edge_sensitivity: normalizeNumber(
      params.edge_sensitivity,
      defaults.edge_sensitivity ?? 0.8,
      0,
      1,
      false,
      2,
    ),
    refine_passes: normalizeNumber(params.refine_passes, defaults.refine_passes ?? 6, 0, 20, true),
    max_merge_color_dist: normalizeNumber(
      params.max_merge_color_dist,
      defaults.max_merge_color_dist ?? 150,
      0,
      2000,
      false,
      0,
    ),
    thin_line_max_radius: normalizeNumber(
      params.thin_line_max_radius,
      defaults.thin_line_max_radius ?? 2.5,
      0.5,
      10,
      false,
      1,
    ),
    min_region_area: normalizeNumber(
      params.min_region_area,
      defaults.min_region_area ?? 10,
      0,
      1000000,
      true,
    ),
    min_contour_area: normalizeNumber(
      params.min_contour_area,
      defaults.min_contour_area ?? 10,
      0,
      1000000,
    ),
    min_hole_area: normalizeNumber(
      params.min_hole_area,
      defaults.min_hole_area ?? 4,
      0,
      1000000,
      false,
      1,
    ),
    contour_simplify: normalizeNumber(
      params.contour_simplify,
      defaults.contour_simplify ?? 0.45,
      0,
      10,
      false,
      2,
    ),
    enable_coverage_fix: normalizeBoolean(
      params.enable_coverage_fix,
      defaults.enable_coverage_fix ?? true,
    ),
    min_coverage_ratio: normalizeNumber(
      params.min_coverage_ratio,
      defaults.min_coverage_ratio ?? 0.998,
      0,
      1,
      false,
      3,
    ),
    svg_enable_stroke: normalizeBoolean(
      params.svg_enable_stroke,
      defaults.svg_enable_stroke ?? true,
    ),
    svg_stroke_width: normalizeNumber(
      params.svg_stroke_width,
      defaults.svg_stroke_width ?? 0.5,
      0,
      20,
      false,
      1,
    ),
  }
}
