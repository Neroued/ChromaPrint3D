import {
  fetchRecipeAlternatives as fetchRecipeAlternativesApi,
  fetchRecipeEditorSummary as fetchRecipeEditorSummaryApi,
  predictRecipeColor as predictRecipeColorApi,
  replaceRecipe as replaceRecipeApi,
  submitGenerateModel as submitGenerateModelApi,
} from '../api/recipeEditor'
import { fetchTaskStatus, getPreviewPath } from '../api/tasks'
import { fetchBlobWithSession } from '../runtime/protectedRequest'
import type { LabColor, RecipeCandidate, RecipeEditorSummary, TaskStatus } from '../types'

const DEFAULT_GENERATION_POLL_INTERVAL_MS = 1000
const DEFAULT_GENERATION_MAX_ATTEMPTS = 300

type SleepFn = (ms: number) => Promise<void>
type FetchStatusFn = (taskId: string) => Promise<TaskStatus>

export type WaitForRecipeGenerationOptions = {
  pollIntervalMs?: number
  maxAttempts?: number
  failedMessage?: string
  timeoutMessage?: string
  sleep?: SleepFn
  fetchStatus?: FetchStatusFn
}

function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export const fetchRecipeEditorSummary = fetchRecipeEditorSummaryApi
export const fetchRecipeAlternatives = fetchRecipeAlternativesApi
export const replaceRecipe = replaceRecipeApi
export const submitGenerateModel = submitGenerateModelApi
export const predictRecipeColor = predictRecipeColorApi

export async function fetchRecipeTaskStatus(taskId: string): Promise<TaskStatus> {
  return fetchTaskStatus(taskId)
}

export async function fetchRecipeEditorPreview(taskId: string): Promise<Blob> {
  return fetchBlobWithSession(getPreviewPath(taskId))
}

export async function waitForRecipeGeneration(
  taskId: string,
  options: WaitForRecipeGenerationOptions = {},
): Promise<TaskStatus> {
  const sleep = options.sleep ?? defaultSleep
  const fetchStatus = options.fetchStatus ?? fetchRecipeTaskStatus
  const pollIntervalMs = options.pollIntervalMs ?? DEFAULT_GENERATION_POLL_INTERVAL_MS
  const maxAttempts = options.maxAttempts ?? DEFAULT_GENERATION_MAX_ATTEMPTS
  const failedMessage = options.failedMessage ?? 'Generation failed'
  const timeoutMessage = options.timeoutMessage ?? 'Generation timed out'

  for (let i = 0; i < maxAttempts; i++) {
    await sleep(pollIntervalMs)
    const status = await fetchStatus(taskId)
    const generationStatus = status.generation?.status ?? 'idle'
    if (generationStatus === 'succeeded') return status
    if (generationStatus === 'failed') {
      throw new Error(status.generation?.error ?? failedMessage)
    }
    if (status.status === 'failed') {
      throw new Error(status.error ?? failedMessage)
    }
  }

  throw new Error(timeoutMessage)
}

export type { LabColor, RecipeCandidate, RecipeEditorSummary }
