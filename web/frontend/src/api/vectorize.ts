import type { VectorizeParams, VectorizeTaskStatus } from '../types'
import { request } from './base'
import { deleteTask } from './tasks'

export async function submitVectorize(
  file: File,
  params: VectorizeParams,
): Promise<{ task_id: string }> {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('params', JSON.stringify(params))
  return request<{ task_id: string }>('/api/v1/vectorize/tasks', {
    method: 'POST',
    body: formData,
  })
}

export async function fetchVectorizeDefaults(): Promise<VectorizeParams> {
  return request<VectorizeParams>('/api/v1/vectorize/defaults')
}

export async function fetchVectorizeTaskStatus(taskId: string): Promise<VectorizeTaskStatus> {
  const raw = await request<Record<string, unknown>>(`/api/v1/tasks/${taskId}`)
  return {
    id: String(raw.id ?? taskId),
    status: String(raw.status ?? 'pending') as VectorizeTaskStatus['status'],
    error: (raw.error as string | null | undefined) ?? null,
    width: Number(raw.width ?? 0),
    height: Number(raw.height ?? 0),
    num_shapes: Number(raw.num_shapes ?? 0),
    svg_size: Number(raw.svg_size ?? 0),
    resolved_num_colors: Number(raw.resolved_num_colors ?? 0),
    timing: (raw.timing as VectorizeTaskStatus['timing']) ?? null,
  }
}

export function getVectorizeSvgPath(id: string): string {
  return `/api/v1/tasks/${id}/artifacts/svg`
}

export async function deleteVectorizeTask(id: string): Promise<void> {
  await deleteTask(id)
}
