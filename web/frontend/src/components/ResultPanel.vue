<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { storeToRefs } from 'pinia'
import {
  NCard,
  NButton,
  NButtonGroup,
  NDescriptions,
  NDescriptionsItem,
  NSlider,
  NSpace,
  NText,
} from 'naive-ui'
import { usePanZoomLinkage } from '../composables/usePanZoomLinkage'
import { useObjectUrlLifecycle } from '../composables/useObjectUrlLifecycle'
import { usePanZoomGroups } from '../composables/usePanZoomGroups'
import ZoomableImageViewport from './common/ZoomableImageViewport.vue'
import {
  getAvailableLayerCount,
  getLayerArtifactKeyFromTop,
  getTopLayerPosition,
} from '../domain/result/layerPreview'
import { fetchBlobWithSession } from '../runtime/protectedRequest'
import { getLayerPreviewPath, getPreviewPath, getSourceMaskPath } from '../services/resultService'
import { useAppStore } from '../stores/app'

const { t } = useI18n()
const appStore = useAppStore()
const { completedTask } = storeToRefs(appStore)
const isCompleted = computed(() => completedTask.value?.status === 'completed')
const result = computed(() => completedTask.value?.result ?? null)
const taskId = computed(() => completedTask.value?.id ?? '')

type ResultImageView = 'preview' | 'source-mask'

const activeImageView = ref<ResultImageView>('preview')
const sliderTopLayerPosition = ref(1)
const visibleLayerArtifactKey = ref<string | null>(null)
const pendingLayerArtifactKey = ref<string | null>(null)
const layerArtifactUrlCache = ref<Record<string, string>>({})
const layerArtifactLoadPromises = new Map<string, Promise<string | null>>()
const { createUrl, revokeUrl } = useObjectUrlLifecycle()
const currentImageBlobUrl = ref('')
const currentImageLoading = ref(false)
let currentImageRequestId = 0

const hasPreview = computed(() => Boolean(result.value?.has_preview))
const hasSourceMask = computed(() => Boolean(result.value?.has_source_mask))
const canToggleResultImage = computed(() => hasPreview.value && hasSourceMask.value)
const layerPreviews = computed(() => result.value?.layer_previews ?? null)
const availableLayerCount = computed(() => getAvailableLayerCount(layerPreviews.value))
const hasLayerPreviews = computed(() => Boolean(taskId.value) && availableLayerCount.value > 0)
const hasCurrentImage = computed(() => currentImageView.value !== null)
const useSingleImageLayout = computed(() => !hasLayerPreviews.value || !hasCurrentImage.value)

const currentImageView = computed<ResultImageView | null>(() => {
  if (activeImageView.value === 'preview') {
    if (hasPreview.value) return 'preview'
    if (hasSourceMask.value) return 'source-mask'
    return null
  }
  if (hasSourceMask.value) return 'source-mask'
  if (hasPreview.value) return 'preview'
  return null
})

const currentImageTitle = computed(() =>
  currentImageView.value === 'source-mask' ? t('result.sourceMask') : t('result.preview'),
)

const currentImageUrl = computed(() => {
  return currentImageBlobUrl.value
})

const showImageSection = computed(() => hasCurrentImage.value || hasLayerPreviews.value)

const currentTopLayerPosition = computed(() =>
  getTopLayerPosition(layerPreviews.value, sliderTopLayerPosition.value),
)

const currentLayerArtifactKey = computed(() =>
  getLayerArtifactKeyFromTop(layerPreviews.value, currentTopLayerPosition.value),
)

const currentLayerImageUrl = computed(() => {
  if (!visibleLayerArtifactKey.value) return ''
  return layerArtifactUrlCache.value[visibleLayerArtifactKey.value] ?? ''
})

const currentLayerTitle = computed(() => {
  if (!hasLayerPreviews.value) return ''
  return `Top ${currentTopLayerPosition.value} / ${availableLayerCount.value}`
})

const currentLayerMeta = computed(() => {
  if (!layerPreviews.value) return ''
  const orderText =
    layerPreviews.value.layer_order === 'Bottom2Top'
      ? t('result.layerOrder.bottomToTop')
      : t('result.layerOrder.topToBottom')
  return `${layerPreviews.value.width}×${layerPreviews.value.height} px | ${orderText}`
})

const isLayerImageLoading = computed(() => {
  const targetArtifact = currentLayerArtifactKey.value
  if (!targetArtifact || !hasLayerPreviews.value) return false
  return visibleLayerArtifactKey.value !== targetArtifact
})

const panZoomGroups = usePanZoomGroups({
  current: 'compare',
  layer: 'compare',
})

usePanZoomLinkage({
  panZoomGroups,
  linkedGroups: {
    current: 'A',
    layer: 'A',
  },
  independentGroups: {
    current: 'self:current',
    layer: 'self:layer',
  },
})

async function loadCurrentImage(view: ResultImageView | null): Promise<void> {
  currentImageRequestId += 1
  const requestId = currentImageRequestId
  currentImageLoading.value = Boolean(view && taskId.value)
  revokeUrl(currentImageBlobUrl.value)
  currentImageBlobUrl.value = ''
  if (!view || !taskId.value) {
    currentImageLoading.value = false
    return
  }

  const requestTaskId = taskId.value
  const imageUrl =
    view === 'source-mask' ? getSourceMaskPath(requestTaskId) : getPreviewPath(requestTaskId)
  try {
    const blob = await fetchBlobWithSession(imageUrl)
    if (currentImageRequestId !== requestId || taskId.value !== requestTaskId) return
    currentImageBlobUrl.value = createUrl(blob)
  } catch {
    // keep the viewport empty when the preview endpoint is not reachable
  } finally {
    if (currentImageRequestId === requestId) {
      currentImageLoading.value = false
    }
  }
}

function clearLayerImageCache() {
  const cachedUrls = Object.values(layerArtifactUrlCache.value)
  for (const url of cachedUrls) revokeUrl(url)
  layerArtifactUrlCache.value = {}
  layerArtifactLoadPromises.clear()
  visibleLayerArtifactKey.value = null
  pendingLayerArtifactKey.value = null
}

async function loadLayerArtifactToCache(artifactKey: string): Promise<string | null> {
  if (!artifactKey || !taskId.value) return null
  const cached = layerArtifactUrlCache.value[artifactKey]
  if (cached) return cached

  const loading = layerArtifactLoadPromises.get(artifactKey)
  if (loading) return loading

  const requestTaskId = taskId.value
  const promise = (async () => {
    try {
      const blob = await fetchBlobWithSession(getLayerPreviewPath(requestTaskId, artifactKey))
      if (taskId.value !== requestTaskId) return null

      const objectUrl = createUrl(blob)
      const current = layerArtifactUrlCache.value[artifactKey]
      if (current) {
        revokeUrl(objectUrl)
        return current
      }
      layerArtifactUrlCache.value = {
        ...layerArtifactUrlCache.value,
        [artifactKey]: objectUrl,
      }
      return objectUrl
    } catch {
      return null
    } finally {
      layerArtifactLoadPromises.delete(artifactKey)
    }
  })()
  layerArtifactLoadPromises.set(artifactKey, promise)
  return promise
}

function prefetchNeighborLayerArtifacts(centerArtifactKey: string | null) {
  const summary = layerPreviews.value
  if (!summary || !centerArtifactKey) return

  const centerIndex = summary.artifacts.indexOf(centerArtifactKey)
  if (centerIndex < 0) return

  const candidates = [
    summary.artifacts[centerIndex - 1],
    summary.artifacts[centerIndex + 1],
  ].filter((item): item is string => Boolean(item))
  for (const candidate of candidates) {
    void loadLayerArtifactToCache(candidate)
  }
}

async function ensureVisibleLayerArtifact(artifactKey: string | null) {
  pendingLayerArtifactKey.value = artifactKey
  if (!artifactKey) {
    visibleLayerArtifactKey.value = null
    return
  }

  const cached = layerArtifactUrlCache.value[artifactKey]
  if (cached) {
    visibleLayerArtifactKey.value = artifactKey
    prefetchNeighborLayerArtifacts(artifactKey)
    return
  }

  const loaded = await loadLayerArtifactToCache(artifactKey)
  if (pendingLayerArtifactKey.value === artifactKey && loaded) {
    visibleLayerArtifactKey.value = artifactKey
  }
  if (pendingLayerArtifactKey.value === artifactKey) {
    prefetchNeighborLayerArtifacts(artifactKey)
  }
}

function handleLayerSliderUpdate(value: number) {
  sliderTopLayerPosition.value = getTopLayerPosition(layerPreviews.value, value)
}

onUnmounted(() => {
  currentImageRequestId += 1
  currentImageLoading.value = false
  revokeUrl(currentImageBlobUrl.value)
  currentImageBlobUrl.value = ''
  clearLayerImageCache()
})

watch(
  () => taskId.value,
  () => {
    // 每次新任务结果默认回到预览图，层位重置到顶部层。
    activeImageView.value = 'preview'
    sliderTopLayerPosition.value = 1
    clearLayerImageCache()
    panZoomGroups.resetAll()
  },
)

watch(
  () => [currentImageView.value, taskId.value] as const,
  ([view]) => {
    void loadCurrentImage(view)
  },
  { immediate: true },
)

watch(
  () => [
    layerPreviews.value?.layers ?? 0,
    layerPreviews.value?.layer_order ?? '',
    layerPreviews.value?.artifacts.length ?? 0,
  ],
  () => {
    sliderTopLayerPosition.value = 1
    clearLayerImageCache()
  },
)

watch(
  () => currentLayerArtifactKey.value,
  (artifactKey) => {
    void ensureVisibleLayerArtifact(artifactKey)
  },
  { immediate: true },
)

function handleSetImageView(view: ResultImageView) {
  if (view === 'preview' && !hasPreview.value) return
  if (view === 'source-mask' && !hasSourceMask.value) return
  activeImageView.value = view
}
</script>

<template>
  <NCard v-if="isCompleted && result" :title="t('result.title')" size="small">
    <template #header-extra>
      <NText v-if="showImageSection" depth="3" class="result-header-tip">
        {{ t('result.zoomHint') }}
      </NText>
    </template>
    <NSpace vertical :size="16">
      <div
        v-if="showImageSection"
        class="result-images-layout"
        :class="{ 'result-images-layout--single': useSingleImageLayout }"
      >
        <NCard
          v-if="currentImageView"
          :title="currentImageTitle"
          size="small"
          embedded
          class="result-image-card"
        >
          <template #header-extra>
            <NSpace :size="8" align="center" class="result-image-card__header-extra">
              <NButtonGroup v-if="canToggleResultImage">
                <NButton
                  size="small"
                  :type="currentImageView === 'preview' ? 'primary' : 'default'"
                  :secondary="currentImageView !== 'preview'"
                  :strong="currentImageView === 'preview'"
                  @click="handleSetImageView('preview')"
                >
                  {{ t('result.preview') }}
                </NButton>
                <NButton
                  size="small"
                  :type="currentImageView === 'source-mask' ? 'primary' : 'default'"
                  :secondary="currentImageView !== 'source-mask'"
                  :strong="currentImageView === 'source-mask'"
                  @click="handleSetImageView('source-mask')"
                >
                  {{ t('result.sourceMask') }}
                </NButton>
              </NButtonGroup>
              <NButton size="tiny" quaternary @click="panZoomGroups.resetAll">{{
                t('result.resetView')
              }}</NButton>
            </NSpace>
          </template>
          <div class="result-image-card__content">
            <ZoomableImageViewport
              :src="currentImageUrl"
              alt="result preview"
              :height="420"
              :controller="panZoomGroups.controllerFor('current')"
            />
            <div class="result-image-card__footer">
              <NText v-if="currentImageLoading" depth="3" class="layer-preview-loading">
                {{ t('result.imageLoading') }}
              </NText>
            </div>
          </div>
        </NCard>

        <NCard
          v-if="hasLayerPreviews"
          :title="t('result.layerPreview')"
          size="small"
          embedded
          class="result-image-card"
        >
          <template #header-extra>
            <NSpace
              :size="10"
              align="center"
              class="result-image-card__header-extra layer-preview-header-meta"
            >
              <NText depth="3" class="layer-preview-title">{{ currentLayerTitle }}</NText>
              <NText depth="3" class="layer-preview-meta-inline">{{ currentLayerMeta }}</NText>
            </NSpace>
          </template>

          <div class="result-image-card__content">
            <div class="layer-preview-body">
              <div class="layer-preview-body__image">
                <ZoomableImageViewport
                  :src="currentLayerImageUrl"
                  alt="layer preview"
                  :height="420"
                  :controller="panZoomGroups.controllerFor('layer')"
                />
              </div>
            </div>

            <div class="layer-preview-controls">
              <NText depth="3" class="layer-preview-slider-label">{{ t('result.layerTop') }}</NText>
              <NSlider
                :value="currentTopLayerPosition"
                :min="1"
                :max="availableLayerCount"
                :step="1"
                class="layer-preview-slider"
                @update:value="handleLayerSliderUpdate"
              />
              <NText depth="3" class="layer-preview-slider-label">{{
                t('result.layerBottom')
              }}</NText>
            </div>

            <div class="result-image-card__footer">
              <NText v-if="isLayerImageLoading" depth="3" class="layer-preview-loading">
                {{ t('result.layerLoading') }}
              </NText>
            </div>
          </div>
        </NCard>
      </div>

      <NText v-if="result.has_3mf" depth="3" style="font-size: 12px">
        {{ result.input_width }}×{{ result.input_height }} px
        <template v-if="result.physical_width_mm > 0">
          | {{ result.physical_width_mm.toFixed(1) }}×{{ result.physical_height_mm.toFixed(1) }}
          mm
        </template>
        <template v-if="result.resolved_pixel_mm > 0">
          | {{ t('result.pixelSize', { value: result.resolved_pixel_mm.toFixed(2) }) }}
        </template>
      </NText>

      <!-- Match statistics -->
      <NDescriptions
        label-placement="left"
        bordered
        :column="2"
        size="small"
        :title="t('result.stats.title')"
      >
        <NDescriptionsItem :label="t('result.stats.totalClusters')">
          {{ result.stats.clusters_total }}
        </NDescriptionsItem>
        <NDescriptionsItem :label="t('result.stats.dbMatch')">
          {{ result.stats.db_only }}
        </NDescriptionsItem>
        <NDescriptionsItem :label="t('result.stats.modelFallback')">
          {{ result.stats.model_fallback }}
        </NDescriptionsItem>
        <NDescriptionsItem :label="t('result.stats.modelQuery')">
          {{ result.stats.model_queries }}
        </NDescriptionsItem>
        <NDescriptionsItem :label="t('result.stats.dbAvgDelta')">
          {{ result.stats.avg_db_de.toFixed(2) }}
        </NDescriptionsItem>
        <NDescriptionsItem :label="t('result.stats.modelAvgDelta')">
          {{ result.stats.avg_model_de.toFixed(2) }}
        </NDescriptionsItem>
      </NDescriptions>
    </NSpace>
  </NCard>
</template>

<style scoped>
.result-header-tip {
  font-size: 11px;
}

.result-images-layout {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  align-items: stretch;
}

.result-images-layout--single {
  grid-template-columns: 1fr;
}

.result-image-card {
  height: 100%;
}

.result-image-card__header-extra {
  min-height: 34px;
}

.result-image-card :deep(.n-card-header) {
  min-height: 56px;
  align-items: center;
}

.result-image-card :deep(.n-card-header__main),
.result-image-card :deep(.n-card-header__extra) {
  display: flex;
  align-items: center;
}

.result-image-card__content {
  display: flex;
  min-height: 100%;
  flex-direction: column;
}

.result-image-card__footer {
  min-height: 28px;
  padding-top: 8px;
}

.layer-preview-title {
  font-size: 12px;
}

.layer-preview-header-meta {
  flex-wrap: wrap;
  justify-content: flex-end;
}

.layer-preview-meta-inline {
  font-size: 12px;
  white-space: nowrap;
}

.layer-preview-body {
  min-width: 0;
}

.layer-preview-body__image {
  min-width: 0;
}

.layer-preview-controls {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 10px;
  margin-top: 12px;
}

.layer-preview-slider {
  width: 100%;
}

.layer-preview-slider-label {
  font-size: 12px;
  white-space: nowrap;
}

.layer-preview-loading {
  display: block;
  margin-top: 8px;
  font-size: 12px;
}

@media (max-width: 960px) {
  .result-images-layout {
    grid-template-columns: 1fr;
  }
}
</style>
