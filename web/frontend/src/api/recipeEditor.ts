import type {
  ConvertRasterParams,
  ConvertVectorParams,
  LabColor,
  RecipeCandidate,
  RecipeEditorSummary,
} from '../types'
import { request } from './base'

export async function submitConvertRasterMatchOnly(
  file: File,
  params: ConvertRasterParams,
): Promise<{ task_id: string }> {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('params', JSON.stringify(params))
  return request<{ task_id: string }>('/api/v1/convert/raster/match-only', {
    method: 'POST',
    body: formData,
  })
}

export async function submitConvertVectorMatchOnly(
  file: File,
  params: ConvertVectorParams,
): Promise<{ task_id: string }> {
  const formData = new FormData()
  formData.append('svg', file)
  formData.append('params', JSON.stringify(params))
  return request<{ task_id: string }>('/api/v1/convert/vector/match-only', {
    method: 'POST',
    body: formData,
  })
}

export async function fetchRecipeEditorSummary(taskId: string): Promise<RecipeEditorSummary> {
  return request<RecipeEditorSummary>(`/api/v1/tasks/${taskId}/recipe-editor/summary`)
}

export async function fetchRecipeAlternatives(
  taskId: string,
  targetLab: LabColor,
  maxCandidates = 10,
  offset = 0,
  recipePattern?: string,
): Promise<RecipeCandidate[]> {
  const payload: Record<string, unknown> = {
    target_lab: targetLab,
    max_candidates: maxCandidates,
    offset,
  }
  if (recipePattern) {
    payload.recipe_pattern = recipePattern
  }
  const data = await request<{ candidates: RecipeCandidate[] }>(
    `/api/v1/tasks/${taskId}/recipe-editor/alternatives`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  )
  return data.candidates
}

export async function replaceRecipe(
  taskId: string,
  regionIds: number[],
  newRecipe: number[],
  newMappedLab: LabColor,
  fromModel: boolean,
): Promise<RecipeEditorSummary> {
  return request<RecipeEditorSummary>(`/api/v1/tasks/${taskId}/recipe-editor/replace`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      region_ids: regionIds,
      new_recipe: newRecipe,
      new_mapped_lab: newMappedLab,
      from_model: fromModel,
    }),
  })
}

export async function submitGenerateModel(taskId: string): Promise<{ task_id: string }> {
  return request<{ task_id: string }>(`/api/v1/tasks/${taskId}/recipe-editor/generate`, {
    method: 'POST',
  })
}

export async function predictRecipeColor(
  taskId: string,
  recipe: number[],
): Promise<{ predicted_lab: LabColor; hex: string; from_model: boolean }> {
  return request<{ predicted_lab: LabColor; hex: string; from_model: boolean }>(
    `/api/v1/tasks/${taskId}/recipe-editor/predict`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipe }),
    },
  )
}

export function getRegionMapPath(taskId: string): string {
  return `/api/v1/tasks/${taskId}/artifacts/region-map`
}
