import { computed, ref, type ComputedRef } from 'vue'
import { useI18n } from 'vue-i18n'
import { toErrorMessage } from '../runtime/error'
import { isElectronRuntime } from '../runtime/platform'
import { resolveErrorCode, shortenError, toDurationMs, trackEvent } from '../services/analytics'

const CONNECTIVITY_CACHE_TTL_MS = 60_000

export function useElectronMattingModels(hasOnlyOpenCvMethod: ComputedRef<boolean>) {
  const { t } = useI18n()
  const isElectron = isElectronRuntime()
  const modelStatus = ref<ElectronModelDownloadStatus | null>(null)
  const modelStatusLoading = ref(false)
  const modelActionLoading = ref(false)
  const modelProgress = ref<ElectronModelDownloadProgress | null>(null)
  const modelError = ref<string | null>(null)
  const modelConnectivity = ref<ElectronModelConnectivityReport | null>(null)
  const modelConnectivityLoading = ref(false)
  const downloadSessionBaseBytes = ref(0)
  const downloadSessionTotalBytes = ref(0)

  const pendingModelCount = computed(
    () => (modelStatus.value?.missingModels ?? 0) + (modelStatus.value?.invalidModels ?? 0),
  )
  const showModelCard = computed(() => {
    if (!isElectron) return false
    if (modelStatus.value?.running) return true
    if (pendingModelCount.value > 0) return true
    if (hasOnlyOpenCvMethod.value) return true
    return Boolean(modelError.value)
  })
  const modelProgressPercent = computed(() => {
    if (modelProgress.value) return modelProgress.value.percent
    const total = modelStatus.value?.totalModels ?? 0
    if (total <= 0) return 0
    return Number((((modelStatus.value?.installedModels ?? 0) / total) * 100).toFixed(1))
  })
  const modelRunning = computed(
    () => Boolean(modelStatus.value?.running) || modelActionLoading.value,
  )
  const modelConnectivitySummary = computed(() => {
    if (!modelConnectivity.value) return ''
    const report = modelConnectivity.value
    return t('matting.connectivity.summary', {
      available: report.availableSources,
      total: report.totalSources,
      models: report.checkedModels,
    })
  })
  const showRestartHint = computed(() => {
    if (!isElectron || !modelStatus.value) return false
    return (
      modelStatus.value.totalModels > 0 &&
      pendingModelCount.value === 0 &&
      hasOnlyOpenCvMethod.value
    )
  })
  const requiredDownloadBytes = computed(() => {
    const models = modelStatus.value?.models ?? []
    return models
      .filter((item) => item.state !== 'installed')
      .reduce((sum, item) => sum + Math.max(0, item.sizeBytes), 0)
  })
  const effectiveDownloadTotalBytes = computed(() => {
    if (downloadSessionTotalBytes.value > 0) return downloadSessionTotalBytes.value
    return requiredDownloadBytes.value
  })
  const downloadedSessionBytes = computed(() => {
    const progress = modelProgress.value
    if (!progress) return 0
    return Math.max(0, progress.downloadedBytes - downloadSessionBaseBytes.value)
  })
  const currentDownloadSpeedBytesPerSec = computed(() => {
    const speed = modelProgress.value?.speedBytesPerSec
    if (typeof speed !== 'number' || !Number.isFinite(speed) || speed <= 0) return null
    return speed
  })

  function formatBytes(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
    const units = ['B', 'KB', 'MB', 'GB', 'TB']
    let value = bytes
    let idx = 0
    while (value >= 1024 && idx < units.length - 1) {
      value /= 1024
      idx += 1
    }
    const fixed = value >= 10 || idx === 0 ? 0 : 1
    return `${value.toFixed(fixed)} ${units[idx]}`
  }

  function formatSpeed(bytesPerSec: number): string {
    return `${formatBytes(bytesPerSec)}/s`
  }

  function formatConnectivityCheckedAt(ts: number): string {
    return new Date(ts).toLocaleTimeString(undefined, { hour12: false })
  }

  function hasFreshConnectivityReport(report: ElectronModelConnectivityReport | null): boolean {
    if (!report) return false
    return Date.now() - report.checkedAtMs <= CONNECTIVITY_CACHE_TTL_MS
  }

  async function checkModelConnectivity(
    force = false,
  ): Promise<ElectronModelConnectivityReport | null> {
    if (!isElectron) return null
    if (!force && hasFreshConnectivityReport(modelConnectivity.value)) {
      return modelConnectivity.value
    }
    const checkConnectivity = window.electron?.models?.checkConnectivity
    if (!checkConnectivity) return null
    modelConnectivityLoading.value = true
    const startTs = performance.now()
    try {
      const report = await checkConnectivity()
      modelConnectivity.value = report
      if (report.availableSources <= 0) {
        modelError.value = t('matting.connectivity.noSource')
      } else if (modelError.value?.includes(t('matting.connectivity.checkLabel'))) {
        modelError.value = null
      }
      trackEvent('matting-connectivity-check', {
        available_sources: report.availableSources,
        total_sources: report.totalSources,
        duration_ms: toDurationMs(performance.now() - startTs),
      })
      return report
    } catch (error: unknown) {
      modelError.value = toErrorMessage(error, t('matting.connectivity.checkFailed'))
      return null
    } finally {
      modelConnectivityLoading.value = false
    }
  }

  async function refreshModelStatus() {
    if (!isElectron) return
    const getStatus = window.electron?.models?.getStatus
    if (!getStatus) return
    modelStatusLoading.value = true
    try {
      modelStatus.value = await getStatus()
      if (modelStatus.value.lastError) {
        modelError.value = modelStatus.value.lastError
      }
    } catch (error: unknown) {
      modelError.value = toErrorMessage(error, t('matting.model.statusFailed'))
    } finally {
      modelStatusLoading.value = false
    }
  }

  let downloadStartTs: number | null = null
  let downloadInitialTotalCount = 0

  function bindModelProgressListener() {
    if (!isElectron) return
    const modelsApi = window.electron?.models
    const onProgress = modelsApi?.onProgress
    if (!onProgress) return
    modelsApi?.clearProgressListener?.()
    onProgress((payload) => {
      modelProgress.value = payload
      if (payload.type === 'start') {
        downloadSessionBaseBytes.value = payload.downloadedBytes
        downloadSessionTotalBytes.value = Math.max(0, payload.totalBytes - payload.downloadedBytes)
      }
      if (payload.type === 'completed' && downloadStartTs !== null) {
        trackEvent('matting-model-download-complete', {
          duration_ms: toDurationMs(performance.now() - downloadStartTs),
          total_count: downloadInitialTotalCount,
        })
        downloadStartTs = null
      }
      if (payload.type === 'error') {
        modelError.value = payload.message
        trackEvent('matting-model-download-fail', {
          error: shortenError(payload.message),
          error_code: resolveErrorCode(payload.message),
        })
        downloadStartTs = null
      }
      if (
        payload.type === 'completed' ||
        payload.type === 'cancelled' ||
        payload.type === 'error'
      ) {
        modelActionLoading.value = false
        void refreshModelStatus()
      }
    })
  }

  async function handleStartModelDownload() {
    const startDownload = window.electron?.models?.startDownload
    if (!startDownload) return
    modelError.value = null
    modelProgress.value = null
    downloadSessionBaseBytes.value = 0
    downloadSessionTotalBytes.value = requiredDownloadBytes.value
    const pendingCount = pendingModelCount.value
    const totalCount = modelStatus.value?.totalModels ?? 0
    downloadStartTs = performance.now()
    downloadInitialTotalCount = totalCount
    trackEvent('matting-model-download-start', {
      pending_count: pendingCount,
      total_count: totalCount,
    })
    try {
      const connectivity = await checkModelConnectivity()
      if (!connectivity || connectivity.availableSources <= 0) {
        throw new Error(t('matting.model.sourceUnreachable'))
      }
      modelActionLoading.value = true
      modelStatus.value = await startDownload()
    } catch (error: unknown) {
      modelError.value = toErrorMessage(error, t('matting.model.downloadFailed'))
      // Pre-download or synchronous failures never reach the progress listener,
      // so we emit the fail event here to guarantee every "start" has a paired
      // terminal event. The progress listener only fires if the IPC task was
      // actually scheduled by the main process.
      if (downloadStartTs !== null) {
        trackEvent('matting-model-download-fail', {
          error: shortenError(error),
          error_code: resolveErrorCode(error),
        })
        downloadStartTs = null
      }
    } finally {
      modelActionLoading.value = false
      await refreshModelStatus()
    }
  }

  async function handleCancelModelDownload() {
    const cancelDownload = window.electron?.models?.cancelDownload
    if (!cancelDownload) return
    trackEvent('matting-model-download-cancel')
    downloadStartTs = null
    try {
      await cancelDownload()
    } catch (error: unknown) {
      modelError.value = toErrorMessage(error, t('matting.model.cancelFailed'))
    }
  }

  async function handleRestartApp() {
    const restartApp = window.electron?.models?.restartApp
    if (!restartApp) return
    try {
      await restartApp()
    } catch (error: unknown) {
      modelError.value = toErrorMessage(error, t('matting.model.restartFailed'))
    }
  }

  function clearModelProgressListener() {
    window.electron?.models?.clearProgressListener?.()
  }

  return {
    bindModelProgressListener,
    checkModelConnectivity,
    clearModelProgressListener,
    currentDownloadSpeedBytesPerSec,
    downloadedSessionBytes,
    effectiveDownloadTotalBytes,
    formatBytes,
    formatConnectivityCheckedAt,
    formatSpeed,
    handleCancelModelDownload,
    handleRestartApp,
    handleStartModelDownload,
    isElectron,
    modelActionLoading,
    modelConnectivity,
    modelConnectivityLoading,
    modelConnectivitySummary,
    modelError,
    modelProgress,
    modelProgressPercent,
    modelRunning,
    modelStatus,
    modelStatusLoading,
    pendingModelCount,
    refreshModelStatus,
    showModelCard,
    showRestartHint,
  }
}
