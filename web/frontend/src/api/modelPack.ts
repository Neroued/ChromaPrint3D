import { request } from './base'

export interface ModelPackMode {
  color_layers: number
  layer_height_mm: number
  num_candidates: number
}

export interface ModelPackScope {
  vendor: string
  material_type: string
}

export interface ModelPackInfo {
  name: string
  schema_version: number
  scope: ModelPackScope
  channel_keys: string[]
  modes: ModelPackMode[]
  defaults: { threshold: number; margin: number }
}

export interface ModelPackInfoResponse {
  packs: ModelPackInfo[]
}

export async function fetchModelPackInfo(): Promise<ModelPackInfoResponse> {
  return request<ModelPackInfoResponse>('/api/v1/model-pack/info')
}
