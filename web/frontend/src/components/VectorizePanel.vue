<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NCollapse,
  NCollapseItem,
  NDivider,
  NForm,
  NFormItem,
  NGrid,
  NGridItem,
  NInputNumber,
  NSpace,
  NSpin,
  NSwitch,
  NText,
  NTooltip,
  NUpload,
  NUploadDragger,
} from 'naive-ui'
import type { UploadFileInfo } from 'naive-ui'
import { useI18n } from 'vue-i18n'
import { useAsyncTask } from '../composables/useAsyncTask'
import { usePanZoomLinkage } from '../composables/usePanZoomLinkage'
import { usePanZoomGroups } from '../composables/usePanZoomGroups'
import { useRasterToolUpload } from '../composables/useRasterToolUpload'
import { useObjectUrlLifecycle } from '../composables/useObjectUrlLifecycle'
import { useBlobDownload } from '../composables/useBlobDownload'
import ZoomableImageViewport from './common/ZoomableImageViewport.vue'
import type { VectorizeParams, VectorizeTaskStatus } from '../types'
import { useAppStore } from '../stores/app'
import {
  defaultVectorizeParams,
  normalizeVectorizeParams,
} from '../domain/vectorize/normalizeVectorizeParams'
import { toErrorMessage } from '../runtime/error'
import { fetchTextWithSession } from '../runtime/protectedRequest'
import {
  fetchVectorizeDefaults,
  fetchVectorizeTaskStatus,
  getVectorizeSvgPath,
  submitVectorize,
} from '../services/vectorizeService'

const { t } = useI18n()

// ── File state ───────────────────────────────────────────────────────────

const svgBlobUrl = ref<string | null>(null)
const { createUrl, revokeUrl } = useObjectUrlLifecycle()
const appStore = useAppStore()
const params = ref<VectorizeParams>({
  ...defaultVectorizeParams,
})
const mergedDefaults = ref<VectorizeParams>({ ...defaultVectorizeParams })

function handleResetParams() {
  params.value = { ...mergedDefaults.value }
}

const upload = useRasterToolUpload({
  onReset: () => {
    resetTask()
    revokeBlobUrls()
    svgContent.value = null
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

const submitParams = computed<VectorizeParams>(() =>
  normalizeVectorizeParams(params.value, defaultVectorizeParams),
)

// ── Task management ─────────────────────────────────────────────────────

const svgContent = ref<string | null>(null)

const {
  taskId,
  status: taskStatus,
  loading,
  error,
  isCompleted,
  submit: submitTask,
  reset: resetTask,
} = useAsyncTask<VectorizeTaskStatus>(
  () => submitVectorize(file.value!, submitParams.value),
  fetchVectorizeTaskStatus,
  {
    onCompleted() {
      fetchSvgBlob(taskId.value!)
    },
  },
)
const { downloadByObjectUrl } = useBlobDownload((message) => {
  error.value = message
})

const canExecute = computed(() => file.value !== null && !loading.value)

async function handleVectorize() {
  if (!file.value) return
  revokeBlobUrls()
  svgContent.value = null
  panZoomGroups.resetAll()
  await submitTask()
}

async function fetchSvgBlob(id: string) {
  try {
    const text = await fetchTextWithSession(getVectorizeSvgPath(id))
    if (taskId.value !== id) return
    svgContent.value = text
    const blob = new Blob([text], { type: 'image/svg+xml' })
    revokeUrl(svgBlobUrl.value)
    svgBlobUrl.value = createUrl(blob)
  } catch (e: unknown) {
    if (taskId.value !== id) return
    error.value = toErrorMessage(e, t('vectorize.fetchFailed'))
  }
}

const panZoomGroups = usePanZoomGroups({
  upload: 'upload',
  original: 'compare',
  result: 'compare',
})

usePanZoomLinkage({
  panZoomGroups,
  linkedGroups: {
    original: 'A',
    result: 'A',
  },
  independentGroups: {
    original: 'self:original',
    result: 'self:result',
  },
})

// ── File management ──────────────────────────────────────────────────────

function revokeBlobUrls() {
  revokeUrl(svgBlobUrl.value)
  svgBlobUrl.value = null
}

// ── Downloads ────────────────────────────────────────────────────────────

async function handleDownloadSvg() {
  if (!svgBlobUrl.value) return
  try {
    await downloadByObjectUrl(svgBlobUrl.value, `${fileBaseName.value}.svg`)
  } catch {
    // handled by useBlobDownload callback
  }
}

function handleUseSvgForConvert() {
  if (!svgContent.value) return
  const resultFile = new File([svgContent.value], `${fileBaseName.value}.svg`, {
    type: 'image/svg+xml',
  })
  appStore.setSelectedFile(resultFile)
  appStore.activeTab = 'convert'
}

// ── Computed helpers ─────────────────────────────────────────────────────

const fileBaseName = computed(() => {
  if (!file.value) return 'image'
  const name = file.value.name
  const dot = name.lastIndexOf('.')
  return dot > 0 ? name.substring(0, dot) : name
})

const timingText = computed(() => {
  const timing = taskStatus.value?.timing
  if (!timing) return null
  const parts: string[] = []
  if (timing.decode_ms > 0)
    parts.push(t('vectorize.timing.decode', { ms: timing.decode_ms.toFixed(0) }))
  if (timing.vectorize_ms > 0)
    parts.push(t('vectorize.timing.vectorize', { ms: timing.vectorize_ms.toFixed(0) }))
  parts.push(t('vectorize.timing.total', { ms: timing.pipeline_ms.toFixed(0) }))
  return parts.join(' | ')
})

const svgSizeText = computed(() => {
  const size = taskStatus.value?.svg_size
  if (!size) return null
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(2)} MB`
})

// ── Lifecycle ────────────────────────────────────────────────────────────

onMounted(async () => {
  try {
    const serverDefaults = await fetchVectorizeDefaults()
    const merged = normalizeVectorizeParams(
      {
        ...params.value,
        ...serverDefaults,
      },
      defaultVectorizeParams,
    )
    params.value = merged
    mergedDefaults.value = { ...merged }
  } catch {
    // Fallback to local defaults when server endpoint is unavailable.
  }
})
</script>

<template>
  <NSpace vertical :size="16">
    <NGrid :cols="2" :x-gap="16" responsive="screen" item-responsive>
      <NGridItem span="2 m:1">
        <NCard :title="t('vectorize.upload.title')" size="small">
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
                  {{ t('vectorize.upload.dropHint') }}
                </NText>
                <NText depth="3" style="font-size: 12px">
                  {{ t('vectorize.upload.formatHint', { formats: rasterImageFormatsText }) }}
                </NText>
                <NText depth="3" style="font-size: 11px">
                  {{
                    t('vectorize.upload.sizeHint', {
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
                {{ t('vectorize.upload.removeImage') }}
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
        <NCard :title="t('vectorize.settings.title')" size="small">
          <template #header-extra>
            <NButton size="tiny" quaternary :disabled="loading" @click="handleResetParams">
              {{ t('common.reset') }}
            </NButton>
          </template>
          <NSpace vertical :size="12">
            <div class="settings-scroll">
              <NText depth="3" style="font-size: 11px">
                {{ t('vectorize.settings.quickTip') }}
              </NText>

              <!-- Core Parameters -->
              <NForm label-placement="left" :label-width="140" :disabled="loading" size="small">
                <NFormItem>
                  <template #label>
                    <NTooltip>
                      <template #trigger>
                        <span class="tip-label">{{ t('vectorize.settings.numColors') }}</span>
                      </template>
                      {{ t('vectorize.settings.numColorsHint') }}
                    </NTooltip>
                  </template>
                  <NInputNumber
                    v-model:value="params.num_colors"
                    :min="2"
                    :max="256"
                    style="width: 100%"
                  />
                </NFormItem>
                <NFormItem>
                  <template #label>
                    <NTooltip>
                      <template #trigger>
                        <span class="tip-label">{{ t('vectorize.settings.curveFitError') }}</span>
                      </template>
                      {{ t('vectorize.settings.curveFitErrorHint') }}
                    </NTooltip>
                  </template>
                  <NInputNumber
                    v-model:value="params.curve_fit_error"
                    :min="0.1"
                    :max="5"
                    :step="0.1"
                    :precision="2"
                    style="width: 100%"
                  />
                </NFormItem>
                <NFormItem>
                  <template #label>
                    <NTooltip>
                      <template #trigger>
                        <span class="tip-label">{{ t('vectorize.settings.contourSimplify') }}</span>
                      </template>
                      {{ t('vectorize.settings.contourSimplifyHint') }}
                    </NTooltip>
                  </template>
                  <NInputNumber
                    v-model:value="params.contour_simplify"
                    :min="0"
                    :max="10"
                    :step="0.05"
                    :precision="2"
                    style="width: 100%"
                  />
                </NFormItem>
              </NForm>

              <!-- Output Enhancement -->
              <NDivider title-placement="left" style="margin: 4px 0; font-size: 12px">
                {{ t('vectorize.settings.outputEnhancement') }}
              </NDivider>

              <NForm label-placement="left" :label-width="140" :disabled="loading" size="small">
                <NFormItem>
                  <template #label>
                    <NTooltip>
                      <template #trigger>
                        <span class="tip-label">{{ t('vectorize.settings.enableStroke') }}</span>
                      </template>
                      {{ t('vectorize.settings.enableStrokeHint') }}
                    </NTooltip>
                  </template>
                  <NSwitch v-model:value="params.svg_enable_stroke" />
                </NFormItem>
                <template v-if="params.svg_enable_stroke">
                  <NFormItem>
                    <template #label>
                      <NTooltip>
                        <template #trigger>
                          <span class="tip-label">{{
                            t('vectorize.advanced.svgStrokeWidth')
                          }}</span>
                        </template>
                        {{ t('vectorize.advanced.svgStrokeWidthHint') }}
                      </NTooltip>
                    </template>
                    <NInputNumber
                      v-model:value="params.svg_stroke_width"
                      :min="0"
                      :max="20"
                      :step="0.1"
                      :precision="1"
                      style="width: 100%"
                    />
                  </NFormItem>
                  <NFormItem>
                    <template #label>
                      <NTooltip>
                        <template #trigger>
                          <span class="tip-label">{{
                            t('vectorize.advanced.thinLineRadius')
                          }}</span>
                        </template>
                        {{ t('vectorize.advanced.thinLineRadiusHint') }}
                      </NTooltip>
                    </template>
                    <NInputNumber
                      v-model:value="params.thin_line_max_radius"
                      :min="0.5"
                      :max="10"
                      :step="0.1"
                      :precision="1"
                      style="width: 100%"
                    />
                  </NFormItem>
                </template>
                <NFormItem>
                  <template #label>
                    <NTooltip>
                      <template #trigger>
                        <span class="tip-label">{{
                          t('vectorize.settings.enableCoverageFix')
                        }}</span>
                      </template>
                      {{ t('vectorize.settings.enableCoverageFixHint') }}
                    </NTooltip>
                  </template>
                  <NSwitch v-model:value="params.enable_coverage_fix" />
                </NFormItem>
                <NFormItem v-if="params.enable_coverage_fix">
                  <template #label>
                    <NTooltip>
                      <template #trigger>
                        <span class="tip-label">{{
                          t('vectorize.advanced.minCoverageRatio')
                        }}</span>
                      </template>
                      {{ t('vectorize.advanced.minCoverageRatioHint') }}
                    </NTooltip>
                  </template>
                  <NInputNumber
                    v-model:value="params.min_coverage_ratio"
                    :min="0"
                    :max="1"
                    :step="0.001"
                    :precision="3"
                    style="width: 100%"
                  />
                </NFormItem>
              </NForm>

              <!-- Advanced Parameters -->
              <NCollapse>
                <NCollapseItem :title="t('vectorize.advanced.title')" name="advanced">
                  <NCollapse>
                    <NCollapseItem
                      :title="t('vectorize.advanced.preprocessing')"
                      name="preprocessing"
                    >
                      <NForm
                        label-placement="left"
                        :label-width="140"
                        :disabled="loading"
                        size="small"
                      >
                        <NFormItem>
                          <template #label>
                            <NTooltip>
                              <template #trigger>
                                <span class="tip-label">{{
                                  t('vectorize.advanced.smoothingSpatial')
                                }}</span>
                              </template>
                              {{ t('vectorize.advanced.smoothingSpatialHint') }}
                            </NTooltip>
                          </template>
                          <NInputNumber
                            v-model:value="params.smoothing_spatial"
                            :min="0"
                            :max="50"
                            :step="0.5"
                            :precision="1"
                            style="width: 100%"
                          />
                        </NFormItem>
                        <NFormItem>
                          <template #label>
                            <NTooltip>
                              <template #trigger>
                                <span class="tip-label">{{
                                  t('vectorize.advanced.smoothingColor')
                                }}</span>
                              </template>
                              {{ t('vectorize.advanced.smoothingColorHint') }}
                            </NTooltip>
                          </template>
                          <NInputNumber
                            v-model:value="params.smoothing_color"
                            :min="0"
                            :max="80"
                            :step="0.5"
                            :precision="1"
                            style="width: 100%"
                          />
                        </NFormItem>
                        <NFormItem>
                          <template #label>
                            <NTooltip>
                              <template #trigger>
                                <span class="tip-label">{{
                                  t('vectorize.advanced.maxWorkingPixels')
                                }}</span>
                              </template>
                              {{ t('vectorize.advanced.maxWorkingPixelsHint') }}
                            </NTooltip>
                          </template>
                          <NInputNumber
                            v-model:value="params.max_working_pixels"
                            :min="0"
                            :max="100000000"
                            :step="100000"
                            style="width: 100%"
                          />
                        </NFormItem>
                      </NForm>
                    </NCollapseItem>

                    <NCollapseItem
                      :title="t('vectorize.advanced.segmentation')"
                      name="segmentation"
                    >
                      <NForm
                        label-placement="left"
                        :label-width="140"
                        :disabled="loading"
                        size="small"
                      >
                        <NFormItem>
                          <template #label>
                            <NTooltip>
                              <template #trigger>
                                <span class="tip-label">{{
                                  t('vectorize.advanced.slicRegionSize')
                                }}</span>
                              </template>
                              {{ t('vectorize.advanced.slicRegionSizeHint') }}
                            </NTooltip>
                          </template>
                          <NInputNumber
                            v-model:value="params.slic_region_size"
                            :min="0"
                            :max="100"
                            style="width: 100%"
                          />
                        </NFormItem>
                        <NFormItem>
                          <template #label>
                            <NTooltip>
                              <template #trigger>
                                <span class="tip-label">{{
                                  t('vectorize.advanced.edgeSensitivity')
                                }}</span>
                              </template>
                              {{ t('vectorize.advanced.edgeSensitivityHint') }}
                            </NTooltip>
                          </template>
                          <NInputNumber
                            v-model:value="params.edge_sensitivity"
                            :min="0"
                            :max="1"
                            :step="0.05"
                            :precision="2"
                            style="width: 100%"
                          />
                        </NFormItem>
                        <NFormItem>
                          <template #label>
                            <NTooltip>
                              <template #trigger>
                                <span class="tip-label">{{
                                  t('vectorize.advanced.refinePasses')
                                }}</span>
                              </template>
                              {{ t('vectorize.advanced.refinePassesHint') }}
                            </NTooltip>
                          </template>
                          <NInputNumber
                            v-model:value="params.refine_passes"
                            :min="0"
                            :max="20"
                            style="width: 100%"
                          />
                        </NFormItem>
                      </NForm>
                    </NCollapseItem>

                    <NCollapseItem :title="t('vectorize.advanced.filtering')" name="filtering">
                      <NForm
                        label-placement="left"
                        :label-width="140"
                        :disabled="loading"
                        size="small"
                      >
                        <NFormItem>
                          <template #label>
                            <NTooltip>
                              <template #trigger>
                                <span class="tip-label">{{
                                  t('vectorize.advanced.mergeColorDist')
                                }}</span>
                              </template>
                              {{ t('vectorize.advanced.mergeColorDistHint') }}
                            </NTooltip>
                          </template>
                          <NInputNumber
                            v-model:value="params.max_merge_color_dist"
                            :min="0"
                            :max="2000"
                            :step="10"
                            style="width: 100%"
                          />
                        </NFormItem>
                        <NFormItem>
                          <template #label>
                            <NTooltip>
                              <template #trigger>
                                <span class="tip-label">{{
                                  t('vectorize.advanced.minRegionArea')
                                }}</span>
                              </template>
                              {{ t('vectorize.advanced.minRegionAreaHint') }}
                            </NTooltip>
                          </template>
                          <NInputNumber
                            v-model:value="params.min_region_area"
                            :min="0"
                            :max="1000000"
                            style="width: 100%"
                          />
                        </NFormItem>
                        <NFormItem>
                          <template #label>
                            <NTooltip>
                              <template #trigger>
                                <span class="tip-label">{{
                                  t('vectorize.advanced.minContourArea')
                                }}</span>
                              </template>
                              {{ t('vectorize.advanced.minContourAreaHint') }}
                            </NTooltip>
                          </template>
                          <NInputNumber
                            v-model:value="params.min_contour_area"
                            :min="0"
                            :max="1000000"
                            style="width: 100%"
                          />
                        </NFormItem>
                        <NFormItem>
                          <template #label>
                            <NTooltip>
                              <template #trigger>
                                <span class="tip-label">{{
                                  t('vectorize.advanced.minHoleArea')
                                }}</span>
                              </template>
                              {{ t('vectorize.advanced.minHoleAreaHint') }}
                            </NTooltip>
                          </template>
                          <NInputNumber
                            v-model:value="params.min_hole_area"
                            :min="0"
                            :max="100000"
                            :step="0.5"
                            :precision="1"
                            style="width: 100%"
                          />
                        </NFormItem>
                      </NForm>
                    </NCollapseItem>
                  </NCollapse>
                </NCollapseItem>
              </NCollapse>
            </div>

            <NButton
              type="primary"
              block
              :loading="loading"
              :disabled="!canExecute"
              @click="handleVectorize"
            >
              {{
                loading ? t('vectorize.actions.processing') : t('vectorize.actions.startVectorize')
              }}
            </NButton>
            <NButton v-if="isCompleted && svgBlobUrl" block @click="handleDownloadSvg">
              {{ t('vectorize.actions.downloadSvg') }}{{ svgSizeText ? ` (${svgSizeText})` : '' }}
            </NButton>
            <NButton
              v-if="isCompleted && svgContent"
              type="success"
              block
              @click="handleUseSvgForConvert"
            >
              {{ t('vectorize.actions.useSvgForConvert') }}
            </NButton>
          </NSpace>
        </NCard>
      </NGridItem>
    </NGrid>

    <NAlert v-if="error" type="error" closable @close="error = null">
      {{ error }}
    </NAlert>

    <div v-if="loading" style="text-align: center; padding: 40px 0">
      <NSpin size="large" />
      <NText depth="3" style="display: block; margin-top: 12px">
        {{
          taskStatus?.status === 'pending'
            ? t('vectorize.status.queuing')
            : t('vectorize.status.running')
        }}
      </NText>
    </div>

    <NCard v-if="isCompleted && svgBlobUrl" :title="t('vectorize.result.title')" size="small">
      <template #header-extra>
        <NSpace :size="8" align="center">
          <NText depth="3" style="font-size: 12px">
            {{ taskStatus!.width }} x {{ taskStatus!.height }} |
            {{ t('vectorize.result.shapes', { count: taskStatus!.num_shapes }) }} | SVG
            {{ svgSizeText }}
          </NText>
          <NButton size="tiny" quaternary @click="panZoomGroups.resetAll">
            {{ t('vectorize.result.resetView') }}
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
        {{ t('vectorize.result.zoomHint') }}
      </NText>
      <div class="preview-row">
        <div class="preview-col">
          <NText depth="3" style="font-size: 12px; margin-bottom: 4px; display: block">
            {{ t('vectorize.result.original') }}
          </NText>
          <ZoomableImageViewport
            :src="originalUrl ?? undefined"
            alt="original"
            :height="400"
            :controller="panZoomGroups.controllerFor('original')"
          />
        </div>
        <div class="preview-col">
          <NText depth="3" style="font-size: 12px; margin-bottom: 4px; display: block">
            {{ t('vectorize.result.svgResult') }}
          </NText>
          <ZoomableImageViewport
            :src="svgBlobUrl ?? undefined"
            alt="svg result"
            :height="400"
            :controller="panZoomGroups.controllerFor('result')"
          />
        </div>
      </div>
    </NCard>
  </NSpace>
</template>

<style scoped>
.settings-scroll {
  max-height: min(520px, 60vh);
  overflow-y: auto;
  padding-right: 4px;
}

.tip-label {
  cursor: help;
  border-bottom: 1px dashed var(--n-text-color-3);
}

.upload-preview {
  text-align: center;
}

.upload-preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.preview-row {
  display: flex;
  gap: 12px;
}

.preview-col {
  flex: 1;
  min-width: 0;
}

@media (max-width: 768px) {
  .preview-row {
    flex-direction: column;
  }
}
</style>
