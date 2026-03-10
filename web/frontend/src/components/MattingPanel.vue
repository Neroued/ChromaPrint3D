<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import {
  NCard,
  NSpace,
  NButton,
  NButtonGroup,
  NSelect,
  NSlider,
  NInputNumber,
  NSwitch,
  NColorPicker,
  NUpload,
  NUploadDragger,
  NText,
  NAlert,
  NSpin,
  NGrid,
  NGridItem,
  NProgress,
} from 'naive-ui'
import type { SelectOption, UploadFileInfo } from 'naive-ui'
import { useI18n } from 'vue-i18n'
import { useAsyncTask } from '../composables/useAsyncTask'
import { usePanZoomLinkage } from '../composables/usePanZoomLinkage'
import { usePanZoomGroups } from '../composables/usePanZoomGroups'
import { useRasterToolUpload } from '../composables/useRasterToolUpload'
import { useObjectUrlLifecycle } from '../composables/useObjectUrlLifecycle'
import { useBlobDownload } from '../composables/useBlobDownload'
import ZoomableImageViewport from './common/ZoomableImageViewport.vue'
import type { MattingMethodInfo, MattingTaskStatus, OutlineMode } from '../types'
import { useAppStore } from '../stores/app'
import { useElectronMattingModels } from '../composables/useElectronMattingModels'
import { toErrorMessage } from '../runtime/error'
import { formatFloat, roundTo } from '../runtime/number'
import { fetchBlobWithSession } from '../runtime/protectedRequest'
import {
  fetchMattingMethods,
  fetchMattingTaskStatus,
  getMattingAlphaPath,
  getMattingForegroundPath,
  getMattingMaskPath,
  getMattingOutlinePath,
  getMattingProcessedForegroundPath,
  getMattingProcessedMaskPath,
  postprocessMatting,
  submitMatting,
} from '../services/mattingService'

const { t } = useI18n()

// ── File state ───────────────────────────────────────────────────────────

const foregroundBlobUrl = ref<string | null>(null)
const foregroundBlob = ref<Blob | null>(null)
const { createUrl, revokeUrl } = useObjectUrlLifecycle()
const appStore = useAppStore()
const upload = useRasterToolUpload({
  onReset: () => {
    resetTask()
    revokeBlobUrls()
    panZoomGroups.resetAll()
  },
  onError: (message) => {
    error.value = message
  },
})
const {
  backendMaxUploadMb,
  clearFile,
  file,
  fileList,
  handleUploadChange,
  imageInfo,
  maxPixelText,
  originalUrl,
  rasterImageAccept,
  rasterImageFormatsText,
} = upload

// ── Postprocess state ────────────────────────────────────────────────────

const threshold = ref(0.5)
const morphCloseSize = ref(0)
const morphCloseIterations = ref(1)
const minRegionArea = ref(0)
const reframeEnabled = ref(false)
const reframePaddingPx = ref(16)
const outlineEnabled = ref(false)
const outlineWidth = ref(2)
const outlineColor = ref('#FFFFFF')
const outlineMode = ref<OutlineMode>('center')
const formatThresholdTooltip = (value: number) => formatFloat(value, 2)

const outlineModeOptions = computed<SelectOption[]>(() => [
  { label: t('matting.strokeCenter'), value: 'center' },
  { label: t('matting.strokeInner'), value: 'inner' },
  { label: t('matting.strokeOuter'), value: 'outer' },
])

const postprocessing = ref(false)
const processedFgBlobUrl = ref<string | null>(null)
const processedFgBlob = ref<Blob | null>(null)
const compositedFgBlobUrl = ref<string | null>(null)
const compositedFgBlob = ref<Blob | null>(null)
const alphaBlobUrl = ref<string | null>(null)
const outlineBlobUrl = ref<string | null>(null)
const thresholdPreviewUrl = ref<string | null>(null)
const thresholdPreviewBlob = ref<Blob | null>(null)
const hasAlpha = computed(() => taskStatus.value?.has_alpha ?? false)
const needsBackendPostprocess = computed(
  () =>
    morphCloseSize.value > 0 ||
    minRegionArea.value > 0 ||
    outlineEnabled.value ||
    reframeEnabled.value,
)
const postprocessPending = computed(
  () => postprocessing.value || (needsBackendPostprocess.value && !processedFgBlob.value),
)
const leftViewMode = ref<'original' | 'alpha'>('original')
const leftPreviewUrl = computed(() =>
  leftViewMode.value === 'alpha' && alphaBlobUrl.value ? alphaBlobUrl.value : originalUrl.value,
)
const currentForegroundUrl = computed(
  () => processedFgBlobUrl.value ?? thresholdPreviewUrl.value ?? foregroundBlobUrl.value,
)
const currentForegroundBlob = computed(
  () => processedFgBlob.value ?? thresholdPreviewBlob.value ?? foregroundBlob.value,
)
const renderedForegroundSize = ref<{ width: number; height: number } | null>(null)
let renderedForegroundSizeRequestId = 0
const renderedForegroundSizeText = computed(() => {
  if (renderedForegroundSize.value) {
    return `${renderedForegroundSize.value.width} x ${renderedForegroundSize.value.height} px`
  }
  const width = taskStatus.value?.width ?? 0
  const height = taskStatus.value?.height ?? 0
  if (width > 0 && height > 0) return `${width} x ${height} px`
  return '--'
})
const exportForegroundBlob = computed(
  () =>
    compositedFgBlob.value ??
    processedFgBlob.value ??
    thresholdPreviewBlob.value ??
    foregroundBlob.value,
)
const exportForegroundUrl = computed(
  () =>
    compositedFgBlobUrl.value ??
    processedFgBlobUrl.value ??
    thresholdPreviewUrl.value ??
    foregroundBlobUrl.value,
)
const OUTLINE_REFERENCE_SHORT_SIDE = 1024
const OUTLINE_MAX_EFFECTIVE_WIDTH = 96
const outlinePreviewShortSide = computed(() => {
  const width = taskStatus.value?.width ?? 0
  const height = taskStatus.value?.height ?? 0
  if (width <= 0 || height <= 0) return 0
  return Math.min(width, height)
})
const outlinePreviewScale = computed(() =>
  Math.max(1, outlinePreviewShortSide.value / OUTLINE_REFERENCE_SHORT_SIDE),
)
const outlineEffectiveWidthPreview = computed(() => {
  const scaled = Math.round(outlineWidth.value * outlinePreviewScale.value)
  return Math.min(OUTLINE_MAX_EFFECTIVE_WIDTH, Math.max(outlineWidth.value, scaled))
})
const outlineAdaptiveHint = computed(() => {
  if (outlinePreviewShortSide.value <= 0) {
    return t('matting.adaptiveHint')
  }
  return t('matting.adaptiveDetail', {
    shortSide: outlinePreviewShortSide.value,
    effectiveWidth: outlineEffectiveWidthPreview.value,
    base: outlineWidth.value,
  })
})

// ── Canvas threshold preview ─────────────────────────────────────────────

let alphaImageData: ImageData | null = null
let originalImageData: ImageData | null = null
let thresholdRafId: number | null = null

function loadImageData(url: string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const c = document.createElement('canvas')
      c.width = img.naturalWidth
      c.height = img.naturalHeight
      const ctx = c.getContext('2d')!
      ctx.drawImage(img, 0, 0)
      resolve(ctx.getImageData(0, 0, c.width, c.height))
    }
    img.onerror = reject
    img.src = url
  })
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = url
  })
}

function compositeOnCanvas(fgUrl: string, overlayUrl: string): Promise<Blob> {
  return Promise.all([loadImage(fgUrl), loadImage(overlayUrl)]).then(([fg, ol]) => {
    const c = document.createElement('canvas')
    c.width = fg.naturalWidth
    c.height = fg.naturalHeight
    const ctx = c.getContext('2d')!
    ctx.drawImage(fg, 0, 0)
    ctx.drawImage(ol, 0, 0)
    return new Promise<Blob>((resolve, reject) => {
      c.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('toBlob failed'))), 'image/png')
    })
  })
}

function applyCanvasThreshold(t: number) {
  if (!alphaImageData || !originalImageData) return
  const aw = alphaImageData.width
  const ah = alphaImageData.height
  if (aw !== originalImageData.width || ah !== originalImageData.height) return

  const c = document.createElement('canvas')
  c.width = aw
  c.height = ah
  const ctx = c.getContext('2d')!
  const output = ctx.createImageData(aw, ah)
  const out = output.data
  const alpha = alphaImageData.data
  const orig = originalImageData.data
  const threshByte = Math.round(t * 255)

  for (let i = 0, len = aw * ah * 4; i < len; i += 4) {
    if ((alpha[i] ?? 0) >= threshByte) {
      out[i] = orig[i] ?? 0
      out[i + 1] = orig[i + 1] ?? 0
      out[i + 2] = orig[i + 2] ?? 0
      out[i + 3] = 255
    }
  }

  ctx.putImageData(output, 0, 0)
  c.toBlob((blob) => {
    if (!blob) return
    revokeUrl(thresholdPreviewUrl.value)
    thresholdPreviewBlob.value = blob
    thresholdPreviewUrl.value = createUrl(blob)
  }, 'image/png')
}

function scheduleCanvasThreshold() {
  if (thresholdRafId !== null) cancelAnimationFrame(thresholdRafId)
  thresholdRafId = requestAnimationFrame(() => {
    thresholdRafId = null
    applyCanvasThreshold(threshold.value)
  })
}

function hexToBgr(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) || 0
  const g = parseInt(hex.slice(3, 5), 16) || 0
  const b = parseInt(hex.slice(5, 7), 16) || 0
  return [b, g, r]
}

const hasEyeDropper = typeof window !== 'undefined' && 'EyeDropper' in window

async function pickColorFromScreen() {
  try {
    const eyeDropper = new (
      window as unknown as { EyeDropper: new () => { open(): Promise<{ sRGBHex: string }> } }
    ).EyeDropper()
    const result = await eyeDropper.open()
    outlineColor.value = result.sRGBHex
  } catch {
    // user cancelled
  }
}

// ── Method selection ─────────────────────────────────────────────────────

const methods = ref<MattingMethodInfo[]>([])
const selectedMethod = ref<string | null>(null)
const methodOptions = computed<SelectOption[]>(() =>
  methods.value.map((m) => ({
    label: m.name,
    value: m.key,
    description: m.description,
  })),
)
const selectedMethodDescription = computed(() => {
  const m = methods.value.find((m) => m.key === selectedMethod.value)
  return m?.description || ''
})
const hasOnlyOpenCvMethod = computed(
  () => methods.value.length > 0 && methods.value.every((m) => m.key === 'opencv'),
)
const {
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
} = useElectronMattingModels(hasOnlyOpenCvMethod)

// ── Task management (composable) ─────────────────────────────────────────

const {
  taskId,
  status: taskStatus,
  loading,
  error,
  isCompleted,
  submit: submitTask,
  reset: resetTask,
} = useAsyncTask<MattingTaskStatus>(
  () => submitMatting(file.value!, selectedMethod.value!),
  fetchMattingTaskStatus,
  {
    onCompleted(status) {
      fetchForegroundBlob(taskId.value!)
      if (status.has_alpha) {
        fetchAlphaBlob(taskId.value!)
      }
      if (needsBackendPostprocess.value) {
        clearBackendResults()
        schedulePostprocess()
      }
    },
  },
)
const { downloadByUrl, downloadByObjectUrl } = useBlobDownload((message) => {
  error.value = message
})

const canExecute = computed(
  () => file.value !== null && selectedMethod.value !== null && !loading.value,
)

async function handleMatting() {
  if (!file.value || !selectedMethod.value) return
  revokeBlobUrls()
  panZoomGroups.resetAll()
  await submitTask()
}

async function fetchForegroundBlob(id: string) {
  try {
    const blob = await fetchBlobWithSession(getMattingForegroundPath(id))
    if (taskId.value !== id) return
    revokeUrl(foregroundBlobUrl.value)
    foregroundBlob.value = blob
    foregroundBlobUrl.value = createUrl(blob)
  } catch (e: unknown) {
    if (taskId.value !== id) return
    error.value = e instanceof Error ? e.message : t('matting.fetchFailed')
  }
}

async function fetchAlphaBlob(id: string) {
  try {
    const blob = await fetchBlobWithSession(getMattingAlphaPath(id))
    if (taskId.value !== id) return
    revokeUrl(alphaBlobUrl.value)
    alphaBlobUrl.value = createUrl(blob)
    const [aData, oData] = await Promise.all([
      loadImageData(alphaBlobUrl.value),
      originalUrl.value ? loadImageData(originalUrl.value) : Promise.resolve(null),
    ])
    if (taskId.value !== id) return
    alphaImageData = aData
    originalImageData = oData
  } catch {
    // alpha is optional, ignore errors
  }
}

// ── Postprocess logic ────────────────────────────────────────────────────

let postprocessTimer: ReturnType<typeof setTimeout> | null = null

function cancelPendingPostprocess() {
  if (postprocessTimer !== null) {
    clearTimeout(postprocessTimer)
    postprocessTimer = null
  }
}

function clearBackendResults() {
  revokeUrl(compositedFgBlobUrl.value)
  compositedFgBlobUrl.value = null
  compositedFgBlob.value = null
  revokeUrl(processedFgBlobUrl.value)
  processedFgBlobUrl.value = null
  processedFgBlob.value = null
  revokeUrl(outlineBlobUrl.value)
  outlineBlobUrl.value = null
}

async function runPostprocess() {
  const id = taskId.value
  if (!id || !isCompleted.value) return
  postprocessing.value = true
  try {
    const result = await postprocessMatting(id, {
      threshold: threshold.value,
      morph_close_size: morphCloseSize.value,
      morph_close_iterations: morphCloseIterations.value,
      min_region_area: minRegionArea.value,
      reframe: {
        enabled: reframeEnabled.value,
        padding_px: reframePaddingPx.value,
      },
      outline: {
        enabled: outlineEnabled.value,
        width: outlineWidth.value,
        color: hexToBgr(outlineColor.value),
        mode: outlineMode.value,
      },
    })
    if (taskId.value !== id) return

    const fgBlob = await fetchBlobWithSession(getMattingProcessedForegroundPath(id))
    if (taskId.value !== id) return
    revokeUrl(processedFgBlobUrl.value)
    processedFgBlob.value = fgBlob
    processedFgBlobUrl.value = createUrl(fgBlob)

    revokeUrl(compositedFgBlobUrl.value)
    compositedFgBlobUrl.value = null
    compositedFgBlob.value = null

    if (result.artifacts.includes('outline')) {
      try {
        const olBlob = await fetchBlobWithSession(getMattingOutlinePath(id))
        if (taskId.value === id) {
          revokeUrl(outlineBlobUrl.value)
          outlineBlobUrl.value = createUrl(olBlob)

          const cBlob = await compositeOnCanvas(processedFgBlobUrl.value!, outlineBlobUrl.value)
          if (taskId.value !== id) return
          compositedFgBlob.value = cBlob
          compositedFgBlobUrl.value = createUrl(cBlob)
        }
      } catch {
        // outline is optional, ignore fetch failures to keep foreground result usable
      }
    } else {
      revokeUrl(outlineBlobUrl.value)
      outlineBlobUrl.value = null
    }
  } catch (e: unknown) {
    if (taskId.value !== id) return
    error.value = e instanceof Error ? e.message : t('matting.postprocessFailed')
  } finally {
    postprocessing.value = false
  }
}

function schedulePostprocess() {
  cancelPendingPostprocess()
  postprocessTimer = setTimeout(runPostprocess, 300)
}

watch(threshold, (next) => {
  const rounded = roundTo(next, 2)
  if (rounded !== next) {
    threshold.value = rounded
    return
  }
  if (!taskId.value || !isCompleted.value || !hasAlpha.value) return
  clearBackendResults()
  scheduleCanvasThreshold()
  if (needsBackendPostprocess.value) {
    schedulePostprocess()
  }
})

watch(
  [
    morphCloseSize,
    morphCloseIterations,
    minRegionArea,
    reframeEnabled,
    reframePaddingPx,
    outlineEnabled,
    outlineWidth,
    outlineColor,
    outlineMode,
  ],
  () => {
    if (!taskId.value || !isCompleted.value) return
    clearBackendResults()
    schedulePostprocess()
  },
)

watch(
  () => currentForegroundUrl.value,
  (url) => {
    renderedForegroundSizeRequestId += 1
    const requestId = renderedForegroundSizeRequestId
    if (!url) {
      renderedForegroundSize.value = null
      return
    }
    void loadImage(url)
      .then((img) => {
        if (renderedForegroundSizeRequestId !== requestId) return
        renderedForegroundSize.value = {
          width: img.naturalWidth,
          height: img.naturalHeight,
        }
      })
      .catch(() => {
        if (renderedForegroundSizeRequestId !== requestId) return
        renderedForegroundSize.value = null
      })
  },
  { immediate: true },
)

onUnmounted(() => {
  cancelPendingPostprocess()
  if (thresholdRafId !== null) cancelAnimationFrame(thresholdRafId)
  clearModelProgressListener()
})

function handleResetPostprocessParams() {
  threshold.value = 0.5
  morphCloseSize.value = 0
  morphCloseIterations.value = 1
  minRegionArea.value = 0
  reframeEnabled.value = false
  reframePaddingPx.value = 16
  outlineEnabled.value = false
  outlineWidth.value = 2
  outlineColor.value = '#FFFFFF'
  outlineMode.value = 'center'
}

function resetPostprocessState() {
  cancelPendingPostprocess()
  if (thresholdRafId !== null) {
    cancelAnimationFrame(thresholdRafId)
    thresholdRafId = null
  }
  revokeUrl(compositedFgBlobUrl.value)
  compositedFgBlobUrl.value = null
  compositedFgBlob.value = null
  revokeUrl(processedFgBlobUrl.value)
  processedFgBlobUrl.value = null
  processedFgBlob.value = null
  revokeUrl(alphaBlobUrl.value)
  alphaBlobUrl.value = null
  revokeUrl(outlineBlobUrl.value)
  outlineBlobUrl.value = null
  revokeUrl(thresholdPreviewUrl.value)
  thresholdPreviewUrl.value = null
  thresholdPreviewBlob.value = null
  renderedForegroundSizeRequestId += 1
  renderedForegroundSize.value = null
  alphaImageData = null
  originalImageData = null
  leftViewMode.value = 'original'
}

const panZoomGroups = usePanZoomGroups({
  upload: 'upload',
  left: 'compare',
  right: 'compare',
})

usePanZoomLinkage({
  panZoomGroups,
  linkedGroups: {
    left: 'A',
    right: 'A',
  },
  independentGroups: {
    left: 'self:left',
    right: 'self:right',
  },
})

// ── File management ──────────────────────────────────────────────────────

function revokeBlobUrls() {
  revokeUrl(foregroundBlobUrl.value)
  foregroundBlobUrl.value = null
  foregroundBlob.value = null
  resetPostprocessState()
}

// ── Downloads ────────────────────────────────────────────────────────────

async function downloadBlob(url: string, filename: string) {
  try {
    await downloadByUrl(url, filename)
  } catch (downloadError: unknown) {
    error.value = toErrorMessage(downloadError, t('matting.downloadFailed'))
  }
}

const fileBaseName = computed(() => {
  if (!file.value) return 'image'
  const name = file.value.name
  const dot = name.lastIndexOf('.')
  return dot > 0 ? name.substring(0, dot) : name
})

function handleDownloadMask() {
  if (!taskId.value || !isCompleted.value) return
  const url = processedFgBlobUrl.value
    ? getMattingProcessedMaskPath(taskId.value)
    : getMattingMaskPath(taskId.value)
  downloadBlob(url, `${fileBaseName.value}_mask.png`)
}

async function handleDownloadForeground() {
  const url = exportForegroundUrl.value
  if (!url) return
  try {
    await downloadByObjectUrl(url, `${fileBaseName.value}_foreground.png`)
  } catch {
    // handled by useBlobDownload callback
  }
}

function handleUseForegroundForConvert() {
  const blob = exportForegroundBlob.value
  if (!blob) return
  const resultFile = new File([blob], `${fileBaseName.value}_foreground.png`, {
    type: blob.type || 'image/png',
  })
  appStore.setSelectedFile(resultFile)
  appStore.activeTab = 'convert'
}

// ── Computed helpers ─────────────────────────────────────────────────────

const timingText = computed(() => {
  const timing = taskStatus.value?.timing
  if (!timing) return null
  const parts: string[] = []
  if (timing.decode_ms > 0)
    parts.push(t('matting.timing.decode', { ms: timing.decode_ms.toFixed(0) }))
  if (timing.preprocess_ms > 0)
    parts.push(t('matting.timing.preprocess', { ms: timing.preprocess_ms.toFixed(0) }))
  if (timing.inference_ms > 0)
    parts.push(t('matting.timing.inference', { ms: timing.inference_ms.toFixed(0) }))
  if (timing.postprocess_ms > 0)
    parts.push(t('matting.timing.postprocess', { ms: timing.postprocess_ms.toFixed(0) }))
  if (timing.encode_ms > 0)
    parts.push(t('matting.timing.encode', { ms: timing.encode_ms.toFixed(0) }))
  parts.push(t('matting.timing.total', { ms: timing.pipeline_ms.toFixed(0) }))
  return parts.join(' | ')
})

// ── Lifecycle ────────────────────────────────────────────────────────────

onMounted(async () => {
  bindModelProgressListener()
  await refreshModelStatus()
  try {
    methods.value = await fetchMattingMethods()
    const first = methods.value[0]
    if (first) {
      selectedMethod.value = first.key
    }
  } catch {
    error.value = t('matting.fetchMethodsFailed')
  }
})
</script>

<template>
  <NSpace vertical :size="16">
    <NCard v-if="showModelCard" :title="t('matting.model.title')" size="small">
      <NSpace vertical :size="8">
        <NText depth="3" style="font-size: 12px">
          {{
            t('matting.model.installedCount', {
              installed: modelStatus?.installedModels ?? 0,
              total: modelStatus?.totalModels ?? 0,
            })
          }}
          <template v-if="pendingModelCount > 0">
            {{
              t('matting.model.pendingCount', {
                pending: pendingModelCount,
                missing: modelStatus?.missingModels ?? 0,
              })
            }}
          </template>
        </NText>
        <NText v-if="modelConnectivitySummary" depth="3" style="font-size: 12px">
          {{ modelConnectivitySummary }}
        </NText>
        <NText v-if="modelConnectivity?.checkedAtMs" depth="3" style="font-size: 11px">
          {{
            t('matting.model.lastCheck', {
              time: formatConnectivityCheckedAt(modelConnectivity.checkedAtMs),
            })
          }}
        </NText>
        <div v-if="modelConnectivity" class="model-connectivity-list">
          <NText
            v-for="source in modelConnectivity.sources"
            :key="`${source.name}-${source.baseUrl}`"
            depth="3"
            style="font-size: 11px; display: block"
          >
            [{{ source.ok ? t('matting.model.available') : t('matting.model.unavailable') }}]
            {{ source.name }} ({{ source.reachableModels }}/{{ source.checkedModels }}) |
            {{ source.responseTimeMs }}ms | {{ source.message }}
          </NText>
        </div>
        <NText depth="3" style="font-size: 12px">
          {{
            t('matting.model.needDownload', {
              need: formatBytes(effectiveDownloadTotalBytes),
              done: formatBytes(downloadedSessionBytes),
            })
          }}
          <template v-if="currentDownloadSpeedBytesPerSec">
            |
            {{ t('matting.model.speed', { speed: formatSpeed(currentDownloadSpeedBytesPerSec) }) }}
          </template>
        </NText>
        <NProgress
          type="line"
          :percentage="modelProgressPercent"
          :processing="modelRunning"
          :show-indicator="true"
        />
        <NText v-if="modelProgress?.message" depth="3" style="font-size: 12px">
          {{ modelProgress.message }}
        </NText>
        <NAlert v-if="showRestartHint" type="warning" :bordered="false">
          {{ t('matting.model.restartHint') }}
        </NAlert>
        <NAlert v-if="modelError" type="error" closable @close="modelError = null">
          {{ modelError }}
        </NAlert>
        <NSpace :size="8">
          <NButton
            type="primary"
            :loading="modelActionLoading"
            :disabled="
              modelRunning ||
              modelStatusLoading ||
              modelConnectivityLoading ||
              pendingModelCount <= 0
            "
            @click="handleStartModelDownload"
          >
            {{ t('matting.model.downloadBtn') }}
          </NButton>
          <NButton
            secondary
            :loading="modelConnectivityLoading"
            :disabled="modelRunning || modelActionLoading || modelStatusLoading"
            @click="checkModelConnectivity(true)"
          >
            {{ t('matting.model.preCheck') }}
          </NButton>
          <NButton
            v-if="modelRunning"
            type="warning"
            secondary
            :disabled="!modelRunning"
            @click="handleCancelModelDownload"
          >
            {{ t('matting.model.cancelDownload') }}
          </NButton>
          <NButton v-if="showRestartHint" type="success" @click="handleRestartApp">
            {{ t('matting.model.restartNow') }}
          </NButton>
          <NButton
            quaternary
            :disabled="modelStatusLoading || modelRunning"
            @click="refreshModelStatus"
          >
            {{ t('matting.model.refreshStatus') }}
          </NButton>
        </NSpace>
        <NText depth="3" style="font-size: 11px">
          {{ t('matting.model.storageTip') }}
        </NText>
      </NSpace>
    </NCard>

    <NGrid :cols="2" :x-gap="16" responsive="screen" item-responsive>
      <NGridItem span="2 m:1">
        <NCard :title="t('matting.upload.title')" size="small">
          <NUpload
            v-if="!file"
            :accept="rasterImageAccept"
            :max="1"
            :default-upload="false"
            :file-list="fileList"
            :show-file-list="false"
            @update:file-list="
              (v: UploadFileInfo[]) => {
                fileList = v
              }
            "
            @change="handleUploadChange"
          >
            <NUploadDragger>
              <NSpace vertical align="center" justify="center" style="padding: 32px 16px">
                <NText depth="3" style="font-size: 14px">
                  {{ t('matting.upload.dropHint') }}
                </NText>
                <NText depth="3" style="font-size: 12px">
                  {{ t('matting.upload.formatHint', { formats: rasterImageFormatsText }) }}
                </NText>
                <NText depth="3" style="font-size: 11px">
                  {{
                    t('matting.upload.sizeHint', {
                      maxMb: backendMaxUploadMb,
                      maxPixels: maxPixelText,
                    })
                  }}
                </NText>
              </NSpace>
            </NUploadDragger>
          </NUpload>
          <div v-else class="upload-preview">
            <div class="upload-preview-header">
              <NText depth="3" style="font-size: 12px">{{ imageInfo }}</NText>
              <NButton size="tiny" quaternary type="error" @click="clearFile">
                {{ t('matting.upload.removeImage') }}
              </NButton>
            </div>
            <ZoomableImageViewport
              :src="originalUrl ?? undefined"
              alt="original"
              :height="260"
              :controller="panZoomGroups.controllerFor('upload')"
            />
          </div>
        </NCard>
      </NGridItem>

      <NGridItem span="2 m:1">
        <NCard :title="t('matting.settings.title')" size="small">
          <NSpace vertical :size="12">
            <div>
              <NText depth="3" style="font-size: 12px; margin-bottom: 4px; display: block">
                {{ t('matting.settings.method') }}
              </NText>
              <NSelect
                v-model:value="selectedMethod"
                :options="methodOptions"
                :placeholder="t('matting.settings.methodPlaceholder')"
                :disabled="loading"
              />
              <NText
                v-if="selectedMethodDescription"
                depth="3"
                style="font-size: 11px; display: block; margin-top: 4px; line-height: 1.5"
              >
                {{ selectedMethodDescription }}
              </NText>
            </div>
            <div>
              <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px">
                <NSwitch v-model:value="reframeEnabled" size="small" :disabled="loading" />
                <NText depth="3" style="font-size: 12px">
                  {{ t('matting.settings.reframe') }}
                </NText>
              </div>
              <NText depth="3" style="font-size: 11px; display: block">
                {{ t('matting.settings.reframeHint') }}
              </NText>
              <div v-if="reframeEnabled" class="slider-input-row" style="margin-top: 8px">
                <NText depth="3" style="font-size: 12px; white-space: nowrap">
                  {{ t('matting.settings.padding') }}
                </NText>
                <NSlider
                  v-model:value="reframePaddingPx"
                  :min="0"
                  :max="512"
                  :step="1"
                  :disabled="loading"
                  class="slider-input-row__slider"
                />
                <NInputNumber
                  v-model:value="reframePaddingPx"
                  :min="0"
                  :max="512"
                  :step="1"
                  :show-button="false"
                  :disabled="loading"
                  class="slider-input-row__input"
                />
              </div>
            </div>
            <NButton
              type="primary"
              block
              :loading="loading"
              :disabled="!canExecute"
              @click="handleMatting"
            >
              {{ loading ? t('matting.actions.processing') : t('matting.actions.startMatting') }}
            </NButton>
            <NButtonGroup v-if="isCompleted && currentForegroundUrl" style="width: 100%">
              <NButton style="flex: 1" :disabled="postprocessPending" @click="handleDownloadMask">
                {{ t('matting.actions.downloadMask') }}
              </NButton>
              <NButton
                style="flex: 1"
                :disabled="postprocessPending"
                @click="handleDownloadForeground"
              >
                {{ t('matting.actions.downloadForeground') }}
              </NButton>
            </NButtonGroup>
            <NButton
              v-if="isCompleted && currentForegroundBlob"
              type="success"
              block
              :disabled="postprocessPending"
              @click="handleUseForegroundForConvert"
            >
              {{ t('matting.actions.useForeground') }}
            </NButton>
          </NSpace>
        </NCard>
      </NGridItem>
    </NGrid>

    <!-- Postprocess controls -->
    <NCard v-if="isCompleted" :title="t('matting.postprocess.title')" size="small">
      <template #header-extra>
        <NSpace :size="8" align="center">
          <NButton size="tiny" quaternary @click="handleResetPostprocessParams">
            {{ t('common.reset') }}
          </NButton>
          <NSpin v-if="postprocessing" :size="16" />
        </NSpace>
      </template>
      <NSpace vertical :size="12">
        <div v-if="hasAlpha">
          <NText depth="3" style="font-size: 12px; margin-bottom: 4px; display: block">
            {{ t('matting.postprocess.threshold') }}
          </NText>
          <div class="slider-input-row">
            <NSlider
              v-model:value="threshold"
              :min="0"
              :max="1"
              :step="0.01"
              :tooltip="true"
              :format-tooltip="formatThresholdTooltip"
              class="slider-input-row__slider"
            />
            <NInputNumber
              v-model:value="threshold"
              :min="0"
              :max="1"
              :step="0.01"
              :precision="2"
              :show-button="false"
              class="slider-input-row__input"
            />
          </div>
        </div>
        <div>
          <NText depth="3" style="font-size: 12px; margin-bottom: 4px; display: block">
            {{ t('matting.postprocess.morphCloseSize') }}
            <NText depth="3" style="font-size: 11px"
              >（{{ t('matting.postprocess.morphCloseSizeHint') }}）</NText
            >
          </NText>
          <div class="slider-input-row">
            <NSlider
              v-model:value="morphCloseSize"
              :min="0"
              :max="51"
              :step="1"
              class="slider-input-row__slider"
            />
            <NInputNumber
              v-model:value="morphCloseSize"
              :min="0"
              :max="51"
              :step="1"
              :show-button="false"
              class="slider-input-row__input"
            />
          </div>
        </div>
        <div v-if="morphCloseSize > 0">
          <NText depth="3" style="font-size: 12px; margin-bottom: 4px; display: block">
            {{ t('matting.postprocess.morphCloseIterations') }}
            <NText depth="3" style="font-size: 11px"
              >（{{ t('matting.postprocess.morphCloseIterationsHint') }}）</NText
            >
          </NText>
          <div class="slider-input-row">
            <NSlider
              v-model:value="morphCloseIterations"
              :min="1"
              :max="10"
              :step="1"
              class="slider-input-row__slider"
            />
            <NInputNumber
              v-model:value="morphCloseIterations"
              :min="1"
              :max="10"
              :step="1"
              :show-button="false"
              class="slider-input-row__input"
            />
          </div>
        </div>
        <div>
          <NText depth="3" style="font-size: 12px; margin-bottom: 4px; display: block">
            {{ t('matting.postprocess.minRegionArea') }}
            <NText depth="3" style="font-size: 11px"
              >（{{ t('matting.postprocess.minRegionAreaHint') }}）</NText
            >
          </NText>
          <div class="slider-input-row">
            <NSlider
              v-model:value="minRegionArea"
              :min="0"
              :max="10000"
              :step="100"
              class="slider-input-row__slider"
            />
            <NInputNumber
              v-model:value="minRegionArea"
              :min="0"
              :max="10000"
              :step="100"
              :show-button="false"
              class="slider-input-row__input"
            />
          </div>
        </div>
        <div>
          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px">
            <NSwitch v-model:value="outlineEnabled" size="small" />
            <NText depth="3" style="font-size: 12px">
              {{ t('matting.postprocess.outline') }}
            </NText>
          </div>
          <template v-if="outlineEnabled">
            <div class="slider-input-row" style="margin-bottom: 8px">
              <NText depth="3" style="font-size: 12px; white-space: nowrap">
                {{ t('matting.postprocess.outlineMode') }}
              </NText>
              <NSelect
                v-model:value="outlineMode"
                :options="outlineModeOptions"
                size="small"
                style="width: 140px"
              />
            </div>
            <div class="slider-input-row">
              <NText depth="3" style="font-size: 12px; white-space: nowrap">
                {{ t('matting.postprocess.outlineWidth') }}
              </NText>
              <NSlider
                v-model:value="outlineWidth"
                :min="1"
                :max="20"
                :step="1"
                class="slider-input-row__slider"
              />
              <NInputNumber
                v-model:value="outlineWidth"
                :min="1"
                :max="20"
                :step="1"
                :show-button="false"
                class="slider-input-row__input"
              />
            </div>
            <NText depth="3" style="font-size: 11px; display: block; margin-top: 4px">
              {{ outlineAdaptiveHint }}
            </NText>
            <div class="slider-input-row">
              <NText depth="3" style="font-size: 12px; white-space: nowrap">
                {{ t('matting.postprocess.outlineColor') }}
              </NText>
              <NColorPicker
                v-model:value="outlineColor"
                :modes="['hex']"
                :show-alpha="false"
                size="small"
                style="width: 120px"
              />
              <NButton
                v-if="hasEyeDropper"
                size="small"
                quaternary
                :title="t('matting.postprocess.eyeDropper')"
                @click="pickColorFromScreen"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <path d="m2 22 1-1h3l9-9" />
                  <path d="M3 21v-3l9-9" />
                  <path
                    d="m15 6 3.4-3.4a2.1 2.1 0 1 1 3 3L18 9l.4.4a2.1 2.1 0 1 1-3 3l-3.8-3.8a2.1 2.1 0 1 1 3-3l.4.4Z"
                  />
                </svg>
              </NButton>
            </div>
          </template>
        </div>
      </NSpace>
    </NCard>

    <NAlert v-if="error" type="error" closable @close="error = null">
      {{ error }}
    </NAlert>

    <div v-if="loading" style="text-align: center; padding: 40px 0">
      <NSpin size="large" />
      <NText depth="3" style="display: block; margin-top: 12px">
        {{
          taskStatus?.status === 'pending'
            ? t('matting.status.queuing')
            : t('matting.status.running')
        }}
      </NText>
    </div>

    <NCard
      v-if="isCompleted && currentForegroundUrl"
      :title="t('matting.result.title')"
      size="small"
    >
      <template #header-extra>
        <NSpace :size="8" align="center">
          <NText depth="3" style="font-size: 12px">
            {{
              t('matting.result.afterMatting', {
                size: renderedForegroundSizeText,
                method: taskStatus!.method,
              })
            }}
          </NText>
          <NButton size="tiny" quaternary @click="panZoomGroups.resetAll">
            {{ t('matting.result.resetView') }}
          </NButton>
        </NSpace>
      </template>
      <NText
        v-if="timingText"
        depth="3"
        style="font-size: 11px; display: block; margin-bottom: 8px; font-family: monospace"
      >
        {{ timingText }}
      </NText>
      <NText depth="3" style="font-size: 11px; display: block; margin-bottom: 8px">
        {{ t('matting.result.zoomHint') }}
      </NText>
      <div class="preview-row">
        <div class="preview-col">
          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px">
            <NText depth="3" style="font-size: 12px">
              {{ leftViewMode === 'alpha' ? 'Alpha' : t('matting.result.original') }}
            </NText>
            <NButtonGroup v-if="hasAlpha" size="tiny">
              <NButton
                :type="leftViewMode === 'original' ? 'primary' : 'default'"
                @click="leftViewMode = 'original'"
              >
                {{ t('matting.result.original') }}
              </NButton>
              <NButton
                :type="leftViewMode === 'alpha' ? 'primary' : 'default'"
                @click="leftViewMode = 'alpha'"
              >
                Alpha
              </NButton>
            </NButtonGroup>
          </div>
          <ZoomableImageViewport
            :src="leftPreviewUrl ?? undefined"
            :alt="leftViewMode"
            :height="400"
            :controller="panZoomGroups.controllerFor('left')"
          />
        </div>
        <div class="preview-col">
          <NText depth="3" style="font-size: 12px; margin-bottom: 4px; display: block">
            {{
              processedFgBlobUrl
                ? t('matting.result.postprocessed')
                : thresholdPreviewUrl
                  ? t('matting.result.thresholdPreview')
                  : t('matting.result.mattingResult')
            }}
          </NText>
          <ZoomableImageViewport
            :src="currentForegroundUrl ?? undefined"
            alt="foreground"
            :height="400"
            :controller="panZoomGroups.controllerFor('right')"
          >
            <template #default="{ transform }">
              <img
                v-if="outlineBlobUrl"
                :src="outlineBlobUrl"
                class="zoomable-image-viewport__layer"
                :style="{ transform }"
                draggable="false"
                alt="outline"
              />
              <div v-if="postprocessing" class="preview-loading">
                <NSpin :size="20" />
              </div>
            </template>
          </ZoomableImageViewport>
        </div>
      </div>
    </NCard>
  </NSpace>
</template>

<style scoped>
.upload-preview {
  text-align: center;
}

.upload-preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.model-connectivity-list {
  max-height: 120px;
  overflow: auto;
  border: 1px dashed var(--n-border-color);
  border-radius: 4px;
  padding: 6px 8px;
}

.preview-row {
  display: flex;
  gap: 12px;
}

.preview-col {
  flex: 1;
  min-width: 0;
}

.slider-input-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.slider-input-row__slider {
  flex: 1;
  min-width: 0;
}

.slider-input-row__input {
  width: 80px;
  flex-shrink: 0;
}

.preview-loading {
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(0, 0, 0, 0.45);
  border-radius: 4px;
  padding: 4px 8px;
  z-index: 1;
}

@media (max-width: 768px) {
  .preview-row {
    flex-direction: column;
  }
}
</style>
