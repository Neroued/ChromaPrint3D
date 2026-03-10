import { ref, computed, onUnmounted, type Ref, type ComputedRef } from 'vue'
import i18n from '../locales'

export interface AsyncTaskOptions<TStatus> {
  pollInterval?: number
  onCompleted?: (status: TStatus) => void
  onFailed?: (status: TStatus) => void
}

export interface AsyncTaskReturn<TStatus> {
  taskId: Ref<string | null>
  status: Ref<TStatus | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  isCompleted: ComputedRef<boolean>
  isFailed: ComputedRef<boolean>
  submit: () => Promise<void>
  reset: () => void
}

/**
 * Generic composable for submit -> poll -> complete async tasks.
 *
 * `loading` is true from submit() until the task reaches completed/failed.
 * There is NO gap in between -- the bug that plagued the old code is
 * structurally impossible here.
 *
 * Race-condition protection: if submit() is called again while a previous
 * task is still polling, the old poll results are silently discarded.
 */
export function useAsyncTask<TStatus extends { status: string; error?: string | null }>(
  submitFn: () => Promise<{ task_id: string }>,
  pollFn: (id: string) => Promise<TStatus>,
  options?: AsyncTaskOptions<TStatus>,
): AsyncTaskReturn<TStatus> {
  const MAX_CONSECUTIVE_POLL_ERRORS = 8

  function isFatalPollError(error: unknown): boolean {
    if (!(error instanceof Error)) return false
    const message = error.message.toLowerCase()
    return (
      message.includes('http 401') ||
      message.includes('http 404') ||
      message.includes('unauthorized') ||
      message.includes('not found')
    )
  }

  const taskId = ref<string | null>(null) as Ref<string | null>
  const status = ref<TStatus | null>(null) as Ref<TStatus | null>
  const loading = ref(false)
  const error = ref<string | null>(null) as Ref<string | null>

  let timer: ReturnType<typeof setInterval> | null = null
  let currentId: string | null = null
  let consecutivePollErrors = 0

  const interval = options?.pollInterval ?? 500

  async function submit() {
    stop()
    currentId = null
    consecutivePollErrors = 0
    loading.value = true
    error.value = null
    taskId.value = null
    status.value = null

    try {
      const resp = await submitFn()
      currentId = resp.task_id
      taskId.value = resp.task_id
      timer = setInterval(poll, interval)
    } catch (e: unknown) {
      currentId = null
      consecutivePollErrors = 0
      loading.value = false
      error.value = e instanceof Error ? e.message : 'Submit failed'
    }
  }

  async function poll() {
    const id = currentId
    if (!id) return

    try {
      const s = await pollFn(id)
      if (currentId !== id) return
      consecutivePollErrors = 0
      status.value = s

      if (s.status === 'completed') {
        stop()
        currentId = null
        loading.value = false
        options?.onCompleted?.(s)
      } else if (s.status === 'failed') {
        stop()
        currentId = null
        loading.value = false
        error.value = s.error || 'Task failed'
        options?.onFailed?.(s)
      }
    } catch (e: unknown) {
      if (currentId !== id) return
      consecutivePollErrors += 1
      const message = e instanceof Error ? e.message : 'Poll failed'
      if (isFatalPollError(e) || consecutivePollErrors >= MAX_CONSECUTIVE_POLL_ERRORS) {
        stop()
        currentId = null
        loading.value = false
        error.value = isFatalPollError(e)
          ? message
          : i18n.global.t('common.pollFailed', { message })
      }
    }
  }

  function stop() {
    if (timer !== null) {
      clearInterval(timer)
      timer = null
    }
  }

  function reset() {
    stop()
    currentId = null
    consecutivePollErrors = 0
    taskId.value = null
    status.value = null
    loading.value = false
    error.value = null
  }

  onUnmounted(stop)

  const isCompleted = computed(() => status.value?.status === 'completed')
  const isFailed = computed(() => status.value?.status === 'failed')

  return { taskId, status, loading, error, isCompleted, isFailed, submit, reset }
}
