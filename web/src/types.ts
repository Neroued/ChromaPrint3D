// ---- Convert parameters (matches backend BuildConvertRequest) ----

export interface ConvertParams {
  db_names?: string[]
  print_mode?: string   // "0.08x5" | "0.04x10"
  color_space?: string  // "lab" | "rgb"
  max_width?: number
  max_height?: number
  scale?: number
  k_candidates?: number
  cluster_count?: number
  allowed_channels?: PaletteChannel[]  // undefined/empty = use all channels
  model_enable?: boolean
  model_only?: boolean
  model_threshold?: number
  model_margin?: number
  flip_y?: boolean
  pixel_mm?: number
  layer_height_mm?: number
  generate_preview?: boolean
  generate_source_mask?: boolean
}

// ---- Match statistics ----

export interface MatchStats {
  clusters_total: number
  db_only: number
  model_fallback: number
  model_queries: number
  avg_db_de: number
  avg_model_de: number
}

// ---- Task result (populated when status === 'completed') ----

export interface TaskResult {
  image_width: number
  image_height: number
  stats: MatchStats
  has_3mf: boolean
  has_preview: boolean
  has_source_mask: boolean
}

// ---- Task status (matches backend TaskInfoToJson) ----

export type TaskStatusValue = 'pending' | 'running' | 'completed' | 'failed'

export type ConvertStage =
  | 'loading_resources'
  | 'processing_image'
  | 'matching'
  | 'building_model'
  | 'exporting'
  | 'unknown'

export interface TaskStatus {
  id: string
  status: TaskStatusValue
  stage: ConvertStage
  progress: number
  created_at_ms: number
  error: string | null
  result: TaskResult | null
}

// ---- ColorDB info (matches backend ColorDBInfoToJson) ----

export interface PaletteChannel {
  color: string
  material: string
}

export interface ColorDBInfo {
  name: string
  num_channels: number
  num_entries: number
  max_color_layers: number
  base_layers: number
  layer_height_mm: number
  line_width_mm: number
  palette: PaletteChannel[]
  source?: 'global' | 'session'
}

// ---- Calibration ----

export interface GenerateBoardRequest {
  palette: PaletteChannel[]
  color_layers?: number
  layer_height_mm?: number
}

export interface GenerateBoardResponse {
  board_id: string
  meta: Record<string, unknown>
}

export interface Generate8ColorBoardRequest {
  palette: PaletteChannel[]
  board_index: number // 1 or 2
}

// ---- Health response ----

export interface HealthResponse {
  status: string
  version: string
  active_tasks: number
  total_tasks: number
}

// ---- Default config response ----

export interface DefaultConfig {
  scale: number
  max_width: number
  max_height: number
  print_mode: string
  color_space: string
  k_candidates: number
  cluster_count: number
  model_enable: boolean
  model_only: boolean
  model_threshold: number
  model_margin: number
  flip_y: boolean
  pixel_mm: number
  layer_height_mm: number
  generate_preview: boolean
  generate_source_mask: boolean
}
