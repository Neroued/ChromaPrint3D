import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useAppStore } from './app'

vi.mock('../api/health', () => ({
  fetchHealth: vi.fn(),
}))

import { fetchHealth } from '../api/health'

describe('app store health state', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('health 请求成功时应视为在线并更新统计', async () => {
    vi.mocked(fetchHealth).mockResolvedValueOnce({
      status: 'degraded',
      version: '1.2.3',
      active_tasks: 2,
      total_tasks: 5,
      memory: {
        rss_bytes: 1,
        heap_allocated_bytes: 2,
        heap_resident_bytes: 3,
        artifact_budget_bytes: 4,
        artifact_budget_limit_bytes: 5,
        colordb_pool_bytes: 6,
        memory_limit_bytes: 7,
        allocator: 'jemalloc',
      },
    })

    const store = useAppStore()
    await store.checkHealth()

    expect(store.serverOnline).toBe(true)
    expect(store.serverVersion).toBe('1.2.3')
    expect(store.activeTasks).toBe(2)
    expect(store.totalTasks).toBe(5)
    expect(store.serverMemory?.allocator).toBe('jemalloc')
  })

  it('health 请求失败时应清空在线状态和旧统计', async () => {
    const store = useAppStore()
    store.serverOnline = true
    store.serverVersion = '1.2.3'
    store.activeTasks = 3
    store.totalTasks = 9
    store.serverMemory = {
      rss_bytes: 1,
      heap_allocated_bytes: 2,
      heap_resident_bytes: 3,
      artifact_budget_bytes: 4,
      artifact_budget_limit_bytes: 5,
      colordb_pool_bytes: 6,
      memory_limit_bytes: 7,
      allocator: 'glibc',
    }

    vi.mocked(fetchHealth).mockRejectedValueOnce(new Error('network error'))

    await store.checkHealth()

    expect(store.serverOnline).toBe(false)
    expect(store.serverVersion).toBe('')
    expect(store.activeTasks).toBe(0)
    expect(store.totalTasks).toBe(0)
    expect(store.serverMemory).toBeNull()
  })
})
