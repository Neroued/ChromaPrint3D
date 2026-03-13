import type {
  ColorDBInfo,
  ConvertRasterParams,
  ConvertVectorParams,
  DefaultConfig,
  WidthAnalysisParams,
  WidthAnalysisResult,
} from '../types'
import { request } from './base'

export async function fetchColorDBs(): Promise<ColorDBInfo[]> {
  const data = await request<{ databases: ColorDBInfo[] }>('/api/v1/databases')
  return data.databases
}

export async function fetchDefaults(): Promise<DefaultConfig> {
  return request<DefaultConfig>('/api/v1/convert/defaults')
}

export async function submitConvertRaster(
  file: File,
  params: ConvertRasterParams,
): Promise<{ task_id: string }> {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('params', JSON.stringify(params))
  return request<{ task_id: string }>('/api/v1/convert/raster', {
    method: 'POST',
    body: formData,
  })
}

export async function submitConvertVector(
  file: File,
  params: ConvertVectorParams,
): Promise<{ task_id: string }> {
  const formData = new FormData()
  formData.append('svg', file)
  formData.append('params', JSON.stringify(params))
  return request<{ task_id: string }>('/api/v1/convert/vector', {
    method: 'POST',
    body: formData,
  })
}

export async function analyzeVectorWidth(
  svg: Blob,
  params: WidthAnalysisParams,
): Promise<WidthAnalysisResult> {
  const formData = new FormData()
  formData.append('svg', svg, 'analyze.svg')
  formData.append('params', JSON.stringify(params))
  return request<WidthAnalysisResult>('/api/v1/convert/vector/analyze-width', {
    method: 'POST',
    body: formData,
  })
}
