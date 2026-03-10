import type {
  ColorDBInfo,
  Generate8ColorBoardRequest,
  GenerateBoardRequest,
  GenerateBoardResponse,
} from '../types'
import { request } from './base'

export async function generateBoard(payload: GenerateBoardRequest): Promise<GenerateBoardResponse> {
  return request<GenerateBoardResponse>('/api/v1/calibration/boards', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function generate8ColorBoard(
  payload: Generate8ColorBoardRequest,
): Promise<GenerateBoardResponse> {
  return request<GenerateBoardResponse>('/api/v1/calibration/boards/8color', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function getBoardModelPath(boardId: string): string {
  return `/api/v1/calibration/boards/${boardId}/model`
}

export function getBoardMetaPath(boardId: string): string {
  return `/api/v1/calibration/boards/${boardId}/meta`
}

export async function buildColorDB(image: File, meta: File, name: string): Promise<ColorDBInfo> {
  const formData = new FormData()
  formData.append('image', image)
  formData.append('meta', meta)
  formData.append('name', name)
  return request<ColorDBInfo>('/api/v1/calibration/colordb', {
    method: 'POST',
    body: formData,
  })
}
