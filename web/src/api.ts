import type {
  HealthResponse,
  ColorDBInfo,
  DefaultConfig,
  ConvertParams,
  TaskStatus,
  GenerateBoardRequest,
  GenerateBoardResponse,
  Generate8ColorBoardRequest,
} from './types'

const BASE = '' // relative — Vite proxy handles /api in dev

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const merged: RequestInit = { credentials: 'include', ...init }
  const res = await fetch(`${BASE}${url}`, merged)
  if (!res.ok) {
    let message = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body.error) message = body.error
    } catch { /* ignore parse errors */ }
    throw new Error(message)
  }
  return res.json() as Promise<T>
}

// ---- Health ----

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health')
}

// ---- ColorDBs ----

export async function fetchColorDBs(): Promise<ColorDBInfo[]> {
  const data = await request<{ databases: ColorDBInfo[] }>('/api/colordbs')
  return data.databases
}

// ---- Default config ----

export async function fetchDefaults(): Promise<DefaultConfig> {
  return request<DefaultConfig>('/api/config/defaults')
}

// ---- Submit conversion ----

export async function submitConvert(
  file: File,
  params: ConvertParams,
): Promise<{ task_id: string }> {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('params', JSON.stringify(params))
  const res = await fetch('/api/convert', { method: 'POST', body: formData, credentials: 'include' })
  if (!res.ok) {
    let message = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body.error) message = body.error
    } catch { /* ignore parse errors */ }
    throw new Error(message)
  }
  return res.json() as Promise<{ task_id: string }>
}

// ---- Task status ----

export async function fetchTaskStatus(id: string): Promise<TaskStatus> {
  return request<TaskStatus>(`/api/tasks/${id}`)
}

// ---- Task list ----

export async function fetchTasks(): Promise<TaskStatus[]> {
  const data = await request<{ tasks: TaskStatus[] }>('/api/tasks')
  return data.tasks
}

// ---- Delete task ----

export async function deleteTask(id: string): Promise<void> {
  await request<{ deleted: boolean }>(`/api/tasks/${id}`, { method: 'DELETE' })
}

// ---- Binary resource URLs (for <img src> or download links) ----

export function getPreviewUrl(id: string): string {
  return `/api/tasks/${id}/preview`
}

export function getSourceMaskUrl(id: string): string {
  return `/api/tasks/${id}/source-mask`
}

export function getResultUrl(id: string): string {
  return `/api/tasks/${id}/result`
}

// ---- Calibration ----

export async function generateBoard(
  payload: GenerateBoardRequest,
): Promise<GenerateBoardResponse> {
  return request<GenerateBoardResponse>('/api/calibration/generate-board', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function generate8ColorBoard(
  payload: Generate8ColorBoardRequest,
): Promise<GenerateBoardResponse> {
  return request<GenerateBoardResponse>('/api/calibration/generate-8color-board', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function getBoardModelUrl(boardId: string): string {
  return `/api/calibration/boards/${boardId}/3mf`
}

export function getBoardMetaUrl(boardId: string): string {
  return `/api/calibration/boards/${boardId}/meta`
}

export async function buildColorDB(
  image: File,
  meta: File,
  name: string,
): Promise<ColorDBInfo> {
  const formData = new FormData()
  formData.append('image', image)
  formData.append('meta', meta)
  formData.append('name', name)
  const res = await fetch('/api/calibration/build-colordb', {
    method: 'POST',
    body: formData,
    credentials: 'include',
  })
  if (!res.ok) {
    let message = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body.error) message = body.error
    } catch { /* ignore parse errors */ }
    throw new Error(message)
  }
  return res.json() as Promise<ColorDBInfo>
}

// ---- Session ColorDBs ----

export async function fetchSessionColorDBs(): Promise<ColorDBInfo[]> {
  const data = await request<{ databases: ColorDBInfo[] }>('/api/session/colordbs')
  return data.databases
}

export async function deleteSessionColorDB(name: string): Promise<void> {
  await request<{ deleted: boolean }>(`/api/session/colordbs/${name}`, { method: 'DELETE' })
}

export function getSessionColorDBDownloadUrl(name: string): string {
  return `/api/session/colordbs/${name}/download`
}

export async function uploadColorDB(
  file: File,
  name?: string,
): Promise<ColorDBInfo> {
  const formData = new FormData()
  formData.append('file', file)
  if (name) formData.append('name', name)
  const res = await fetch('/api/session/colordbs/upload', {
    method: 'POST',
    body: formData,
    credentials: 'include',
  })
  if (!res.ok) {
    let message = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body.error) message = body.error
    } catch { /* ignore parse errors */ }
    throw new Error(message)
  }
  return res.json() as Promise<ColorDBInfo>
}
