import type {
  CalibrationLocateResult,
  ColorDBBuildResult,
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

export async function locateCalibrationBoard(
  image: File,
  meta: File,
): Promise<CalibrationLocateResult> {
  const formData = new FormData()
  formData.append('image', image)
  formData.append('meta', meta)
  return request<CalibrationLocateResult>('/api/v1/calibration/locate', {
    method: 'POST',
    body: formData,
  })
}

export async function buildColorDB(
  image: File,
  meta: File,
  name: string,
  corners?: [number, number][],
): Promise<ColorDBBuildResult> {
  const formData = new FormData()
  formData.append('image', image)
  formData.append('meta', meta)
  formData.append('name', name)
  if (corners) {
    formData.append('corners', JSON.stringify(corners))
  }
  return request<ColorDBBuildResult>('/api/v1/calibration/colordb', {
    method: 'POST',
    body: formData,
  })
}
