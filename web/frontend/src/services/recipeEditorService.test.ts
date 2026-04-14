import { describe, expect, it, vi } from 'vitest'
import { waitForRecipeGeneration } from './recipeEditorService'
import type { TaskStatus } from '../types'

function makeTaskStatus(overrides: Partial<TaskStatus> & Pick<TaskStatus, 'status'>): TaskStatus {
  const { status, ...rest } = overrides
  return {
    id: 'task-1',
    status,
    stage: 'building_model',
    progress: 0,
    created_at_ms: 0,
    error: null,
    result: null,
    ...rest,
  }
}

describe('waitForRecipeGeneration', () => {
  it('在 generation 成功后返回最新任务状态', async () => {
    const fetchStatus = vi
      .fn<(_: string) => Promise<TaskStatus>>()
      .mockResolvedValueOnce(
        makeTaskStatus({
          status: 'completed',
          generation: { status: 'running', error: null },
        }),
      )
      .mockResolvedValueOnce(
        makeTaskStatus({
          status: 'completed',
          generation: { status: 'succeeded', error: null },
          result: {
            input_width: 10,
            input_height: 10,
            resolved_pixel_mm: 0.1,
            physical_width_mm: 1,
            physical_height_mm: 1,
            stats: {
              clusters_total: 1,
              db_only: 1,
              model_fallback: 0,
              model_queries: 0,
              avg_db_de: 0,
              avg_model_de: 0,
            },
            has_3mf: true,
            has_preview: true,
            has_source_mask: true,
          },
        }),
      )

    const status = await waitForRecipeGeneration('task-1', {
      pollIntervalMs: 0,
      maxAttempts: 5,
      sleep: async () => {},
      fetchStatus,
    })

    expect(fetchStatus).toHaveBeenCalledTimes(2)
    expect(status.generation?.status).toBe('succeeded')
    expect(status.result?.has_3mf).toBe(true)
  })

  it('在 generation 失败时抛出 generation.error', async () => {
    const fetchStatus = vi.fn<(_: string) => Promise<TaskStatus>>().mockResolvedValue(
      makeTaskStatus({
        status: 'completed',
        generation: { status: 'failed', error: 'model export failed' },
      }),
    )

    await expect(
      waitForRecipeGeneration('task-1', {
        pollIntervalMs: 0,
        maxAttempts: 1,
        sleep: async () => {},
        fetchStatus,
        failedMessage: 'fallback failed',
      }),
    ).rejects.toThrow('model export failed')
  })

  it('在顶层任务失败时抛出顶层错误', async () => {
    const fetchStatus = vi.fn<(_: string) => Promise<TaskStatus>>().mockResolvedValue(
      makeTaskStatus({
        status: 'failed',
        error: 'task crashed',
        generation: { status: 'running', error: null },
      }),
    )

    await expect(
      waitForRecipeGeneration('task-1', {
        pollIntervalMs: 0,
        maxAttempts: 1,
        sleep: async () => {},
        fetchStatus,
        failedMessage: 'fallback failed',
      }),
    ).rejects.toThrow('task crashed')
  })
})
