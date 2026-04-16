<script setup lang="ts">
import { computed, ref, watch, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  NCard,
  NButton,
  NSpace,
  NText,
  NAlert,
  NSwitch,
  NTooltip,
  NTabs,
  NTabPane,
  useMessage,
} from 'naive-ui'
import type { RecipeEditorSummary, RecipeCandidate, LabColor, RecipeInfo } from '../../types'
import {
  fetchRecipeEditorPreview,
  fetchRecipeEditorSummary,
  fetchRecipeTaskStatus,
  replaceRecipe,
  submitGenerateModel,
  waitForRecipeGeneration,
} from '../../services/recipeEditorService'
import { useRegionMap } from '../../composables/useRegionMap'
import { usePanZoom } from '../../composables/usePanZoom'
import { useAppStore } from '../../stores/app'
import RecipeSummaryPanel from './RecipeSummaryPanel.vue'
import RecipeCandidatePanel from './RecipeCandidatePanel.vue'
import CustomRecipeDialog from './CustomRecipeDialog.vue'
import CustomRecipeListPanel from './CustomRecipeListPanel.vue'
import RegionOverlayCanvas from './RegionOverlayCanvas.vue'
import ZoomableImageViewport from '../common/ZoomableImageViewport.vue'
import { useBlobDownload } from '../../composables/useBlobDownload'
import { getResultPath } from '../../services/resultService'

const props = defineProps<{
  taskId: string
}>()

const { t } = useI18n()
const message = useMessage()
const appStore = useAppStore()
const { downloadByUrl } = useBlobDownload()

const summary = ref<RecipeEditorSummary | null>(null)
const summaryLoading = ref(false)
const summaryError = ref<string | null>(null)
const selectedRegionIds = ref<Set<number>>(new Set())
const selectedRecipeIndex = ref<number | null>(null)
const globalMode = ref(false)
const replacing = ref(false)
const generating = ref(false)
const generateError = ref<string | null>(null)
const generateDone = ref(false)
const editedAfterGenerate = ref(false)
const downloading3mf = ref(false)
const previewBlobUrl = ref('')

interface UndoEntry {
  regionIds: number[]
  oldRecipe: RecipeInfo
  oldRecipeIndex: number
}
const undoStack = ref<UndoEntry[]>([])

const panZoom = usePanZoom()
const previewAreaRef = ref<HTMLElement | null>(null)
const mouseDownPos = ref<{ x: number; y: number } | null>(null)
const hoverInfo = ref<{
  px: number
  py: number
  regionId: number | null
  recipeLabel: string | null
  recipeHex: string | null
} | null>(null)

const isViewTransformed = computed(
  () =>
    panZoom.scale.value !== 1 || panZoom.translateX.value !== 0 || panZoom.translateY.value !== 0,
)

const {
  regionMap,
  load: loadRegionMap,
  getRegionAtPixel,
  getRegionIdsForRecipeIndex,
  clear: clearRegionMap,
} = useRegionMap()

const targetLab = computed<LabColor | null>(() => {
  if (selectedRecipeIndex.value === null || !summary.value) return null
  const recipe = summary.value.unique_recipes[selectedRecipeIndex.value]
  return recipe?.mapped_lab ?? null
})

const targetHex = computed<string | null>(() => {
  if (selectedRecipeIndex.value === null || !summary.value) return null
  const recipe = summary.value.unique_recipes[selectedRecipeIndex.value]
  return recipe?.hex ?? null
})

const regionLookup = computed(() => {
  const map = new Map<number, number>()
  if (!summary.value) return map
  for (const reg of summary.value.regions) {
    map.set(reg.region_id, reg.recipe_index)
  }
  return map
})

// ── Data loading ─────────────────────────────────────────────────────────────

async function loadSummary() {
  summaryLoading.value = true
  summaryError.value = null
  try {
    summary.value = await fetchRecipeEditorSummary(props.taskId)
    await loadRegionMap(props.taskId, summary.value.width, summary.value.height)
    await loadPreview()
    await syncCompletedTask()
    startKeepalive()
  } catch (e) {
    summaryError.value = e instanceof Error ? e.message : String(e)
  } finally {
    summaryLoading.value = false
  }
}

async function loadPreview() {
  if (previewBlobUrl.value) URL.revokeObjectURL(previewBlobUrl.value)
  try {
    const blob = await fetchRecipeEditorPreview(props.taskId)
    previewBlobUrl.value = URL.createObjectURL(blob)
  } catch {
    previewBlobUrl.value = ''
  }
}

async function syncCompletedTask() {
  try {
    const status = await fetchRecipeTaskStatus(props.taskId)
    appStore.setCompletedTask(status)
  } catch {
    appStore.clearCompletedTask()
  }
}

// ── Coordinate mapping (screen → image pixel) ───────────────────────────────

const VIEWPORT_BORDER = 1

function screenToImagePixel(clientX: number, clientY: number): { x: number; y: number } | null {
  const el = previewAreaRef.value
  if (!el || !summary.value) return null

  const rect = el.getBoundingClientRect()
  const contentX = clientX - rect.left - VIEWPORT_BORDER
  const contentY = clientY - rect.top - VIEWPORT_BORDER
  const containerW = rect.width - VIEWPORT_BORDER * 2
  const containerH = rect.height - VIEWPORT_BORDER * 2

  const s = panZoom.scale.value
  const tx = panZoom.translateX.value
  const ty = panZoom.translateY.value
  const localX = (contentX - tx) / s
  const localY = (contentY - ty) / s

  const naturalW = summary.value.width
  const naturalH = summary.value.height
  const fitScale = Math.min(containerW / naturalW, containerH / naturalH)
  const renderedW = naturalW * fitScale
  const renderedH = naturalH * fitScale
  const offsetX = (containerW - renderedW) / 2
  const offsetY = (containerH - renderedH) / 2

  const px = Math.floor((localX - offsetX) / fitScale)
  const py = Math.floor((localY - offsetY) / fitScale)

  if (px < 0 || px >= naturalW || py < 0 || py >= naturalH) return null
  return { x: px, y: py }
}

// ── Hover info ───────────────────────────────────────────────────────────────

function handlePreviewMouseMove(event: MouseEvent) {
  if (!regionMap.value || !summary.value) {
    hoverInfo.value = null
    return
  }
  const pixel = screenToImagePixel(event.clientX, event.clientY)
  if (!pixel) {
    hoverInfo.value = null
    return
  }
  const rid = getRegionAtPixel(pixel.x, pixel.y)
  let recipeLabel: string | null = null
  let recipeHex: string | null = null
  if (rid !== null && rid !== 0xffffffff) {
    const recipeIdx = regionLookup.value.get(rid)
    if (recipeIdx !== undefined) {
      const recipe = summary.value.unique_recipes[recipeIdx]
      if (recipe) {
        recipeLabel = recipe.recipe.join('-')
        recipeHex = recipe.hex
      }
    }
  }
  hoverInfo.value = {
    px: pixel.x,
    py: pixel.y,
    regionId: rid !== null && rid !== 0xffffffff ? rid : null,
    recipeLabel,
    recipeHex,
  }
}

function handlePreviewMouseLeave() {
  hoverInfo.value = null
}

// ── Click / drag handling ────────────────────────────────────────────────────

function recordMouseDown(event: MouseEvent) {
  mouseDownPos.value = { x: event.clientX, y: event.clientY }
}

function handleViewportClick(event: MouseEvent) {
  if (!regionMap.value || !summary.value) return

  if (mouseDownPos.value) {
    const dx = event.clientX - mouseDownPos.value.x
    const dy = event.clientY - mouseDownPos.value.y
    if (dx * dx + dy * dy > 25) return
  }

  const pixel = screenToImagePixel(event.clientX, event.clientY)
  if (!pixel) {
    clearSelection()
    return
  }

  const regionId = getRegionAtPixel(pixel.x, pixel.y)
  if (regionId === null || regionId === 0xffffffff) {
    clearSelection()
    return
  }

  if (globalMode.value) {
    const recipeIdx = regionLookup.value.get(regionId)
    if (recipeIdx !== undefined) {
      if (recipeIdx === selectedRecipeIndex.value) {
        clearSelection()
      } else {
        selectRecipeByIndex(recipeIdx)
      }
    }
  } else {
    const clickedRecipeIdx = regionLookup.value.get(regionId)
    if (clickedRecipeIdx === undefined) return

    if (selectedRecipeIndex.value !== null && clickedRecipeIdx !== selectedRecipeIndex.value) {
      selectedRegionIds.value = new Set([regionId])
      selectedRecipeIndex.value = clickedRecipeIdx
    } else {
      const newSet = new Set(selectedRegionIds.value)
      if (newSet.has(regionId)) {
        newSet.delete(regionId)
      } else {
        newSet.add(regionId)
      }
      selectedRegionIds.value = newSet
      selectedRecipeIndex.value = newSet.size > 0 ? clickedRecipeIdx : null
    }
  }
}

function clearSelection() {
  selectedRegionIds.value = new Set()
  selectedRecipeIndex.value = null
}

const hasSelection = computed(() => selectedRegionIds.value.size > 0)

// ── Recipe selection ─────────────────────────────────────────────────────────

function selectRecipeByIndex(index: number) {
  selectedRecipeIndex.value = index
  if (!summary.value) return
  const ids = getRegionIdsForRecipeIndex(index, summary.value.region_recipe_indices)
  selectedRegionIds.value = new Set(ids)
}

function handleSelectRecipe(index: number) {
  selectRecipeByIndex(index)
}

// ── Replace recipe ───────────────────────────────────────────────────────────

async function handleCandidateSelect(candidate: RecipeCandidate) {
  if (replacing.value || generating.value) return
  if (!summary.value || selectedRegionIds.value.size === 0 || selectedRecipeIndex.value === null) {
    message.warning(t('recipeEditor.noRegionSelected'))
    return
  }
  replacing.value = true
  try {
    const regionIds = Array.from(selectedRegionIds.value)
    const oldRecipeIndex = selectedRecipeIndex.value!
    const srcRecipe = summary.value.unique_recipes[oldRecipeIndex]
    if (!srcRecipe) return
    const oldRecipe: RecipeInfo = { ...srcRecipe }

    const newSummary = await replaceRecipe(
      props.taskId,
      regionIds,
      candidate.recipe,
      candidate.predicted_lab,
      candidate.from_model,
    )

    undoStack.value.push({ regionIds, oldRecipe, oldRecipeIndex })

    summary.value = newSummary
    await loadPreview()
    await syncCompletedTask()

    const selectedSet = selectedRegionIds.value
    const firstRegion = newSummary.regions.find((r) => selectedSet.has(r.region_id))
    if (firstRegion !== undefined) {
      selectedRecipeIndex.value = firstRegion.recipe_index
    }

    if (generateDone.value) editedAfterGenerate.value = true
    window.umami?.track('recipe-replace', { regionCount: regionIds.length })
    message.success(t('recipeEditor.replaceSuccess'))
  } catch (e) {
    message.error(e instanceof Error ? e.message : String(e))
  } finally {
    replacing.value = false
  }
}

async function handleUndo() {
  if (replacing.value || generating.value) return
  if (undoStack.value.length === 0 || !summary.value) return
  const entry = undoStack.value.pop()!
  replacing.value = true
  try {
    const newSummary = await replaceRecipe(
      props.taskId,
      entry.regionIds,
      entry.oldRecipe.recipe,
      entry.oldRecipe.mapped_lab,
      entry.oldRecipe.from_model,
    )
    summary.value = newSummary
    await loadPreview()
    await syncCompletedTask()

    const undoSet = new Set(entry.regionIds)
    const firstRegion = newSummary.regions.find((r) => undoSet.has(r.region_id))
    if (firstRegion !== undefined) {
      selectedRecipeIndex.value = firstRegion.recipe_index
      selectedRegionIds.value = new Set(entry.regionIds)
    } else {
      selectedRecipeIndex.value = null
      selectedRegionIds.value = new Set()
    }

    if (generateDone.value) editedAfterGenerate.value = true
    message.success(t('recipeEditor.replaceSuccess'))
  } catch (e) {
    message.error(e instanceof Error ? e.message : String(e))
  } finally {
    replacing.value = false
  }
}

const canUndo = computed(() => undoStack.value.length > 0)

// ── Custom recipe tab & history ──────────────────────────────────────────────

const rightTab = ref<'alternatives' | 'custom'>('alternatives')
const customRecipes = ref<RecipeCandidate[]>([])

// ── Custom recipe dialog ────────────────────────────────────────────────────

const showCustomRecipeDialog = ref(false)

const customRecipeInitialRecipe = computed<number[]>(() => {
  if (selectedRecipeIndex.value === null || !summary.value) return []
  const recipe = summary.value.unique_recipes[selectedRecipeIndex.value]
  return recipe?.recipe ?? []
})

const customRecipeInitialHex = computed<string>(() => {
  if (selectedRecipeIndex.value === null || !summary.value) return '#FFFFFF'
  const recipe = summary.value.unique_recipes[selectedRecipeIndex.value]
  return recipe?.hex ?? '#FFFFFF'
})

function openCustomRecipeDialog() {
  if (replacing.value || generating.value) return
  if (!summary.value || selectedRegionIds.value.size === 0 || selectedRecipeIndex.value === null) {
    message.warning(t('recipeEditor.noRegionSelected'))
    return
  }
  showCustomRecipeDialog.value = true
}

async function handleCustomRecipeConfirm(payload: {
  recipe: number[]
  mappedLab: import('../../types').LabColor
  hex: string
  fromModel: boolean
}) {
  const candidate: RecipeCandidate = {
    recipe: payload.recipe,
    predicted_lab: payload.mappedLab,
    hex: payload.hex,
    delta_e76: 0,
    lightness_diff: 0,
    hue_diff: 0,
    from_model: payload.fromModel,
  }

  const key = candidate.recipe.join('-')
  const existIdx = customRecipes.value.findIndex((c) => c.recipe.join('-') === key)
  if (existIdx >= 0) {
    customRecipes.value.splice(existIdx, 1)
  }
  customRecipes.value.unshift(candidate)

  rightTab.value = 'custom'
  await handleCandidateSelect(candidate)
}

// ── Generate model ───────────────────────────────────────────────────────────

async function handleGenerate() {
  generating.value = true
  generateError.value = null
  generateDone.value = false
  editedAfterGenerate.value = false
  appStore.clearCompletedTask()
  try {
    await submitGenerateModel(props.taskId)
    const status = await waitForRecipeGeneration(props.taskId, {
      failedMessage: t('recipeEditor.generateFailed'),
      timeoutMessage: t('recipeEditor.generateTimeout'),
    })
    appStore.setCompletedTask(status)
    generateDone.value = true
    window.umami?.track('generate-model-complete')
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    generateError.value = msg
    window.umami?.track('generate-model-fail', { error: msg.slice(0, 120) })
  } finally {
    generating.value = false
  }
}

async function handleDownload3MF() {
  if (downloading3mf.value) return
  downloading3mf.value = true
  try {
    const baseName = appStore.selectedFile?.name?.replace(/\.[^.]+$/, '') ?? 'result'
    await downloadByUrl(getResultPath(props.taskId), `${baseName}.3mf`)
  } catch {
    /* handled by runtime */
  } finally {
    downloading3mf.value = false
  }
}

// ── Keep-alive heartbeat ─────────────────────────────────────────────────────

const KEEPALIVE_INTERVAL_MS = 5 * 60 * 1000

let keepaliveTimer: ReturnType<typeof setInterval> | null = null

function startKeepalive() {
  stopKeepalive()
  keepaliveTimer = setInterval(async () => {
    if (!props.taskId || !summary.value) return
    try {
      summary.value = await fetchRecipeEditorSummary(props.taskId)
    } catch {
      /* silent – don't disrupt the user on heartbeat failure */
    }
  }, KEEPALIVE_INTERVAL_MS)
}

function stopKeepalive() {
  if (keepaliveTimer) {
    clearInterval(keepaliveTimer)
    keepaliveTimer = null
  }
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

watch(
  () => props.taskId,
  () => {
    summary.value = null
    selectedRegionIds.value = new Set()
    selectedRecipeIndex.value = null
    undoStack.value = []
    generateDone.value = false
    editedAfterGenerate.value = false
    stopKeepalive()
    clearRegionMap()
    panZoom.resetView()
    if (props.taskId) void loadSummary()
  },
  { immediate: true },
)

onUnmounted(() => {
  stopKeepalive()
})
</script>

<template>
  <NCard :title="t('recipeEditor.title')" size="small">
    <template #header-extra>
      <NSpace :size="12" align="center">
        <NTooltip>
          <template #trigger>
            <NSpace :size="6" align="center">
              <NText depth="3" style="font-size: 12px">
                {{ t('recipeEditor.globalModeLabel') }}
              </NText>
              <NSwitch v-model:value="globalMode" size="small" />
            </NSpace>
          </template>
          {{ t('recipeEditor.globalModeTooltip') }}
        </NTooltip>
        <NButton
          size="small"
          quaternary
          :style="hasSelection ? undefined : { visibility: 'hidden', pointerEvents: 'none' }"
          @click="clearSelection"
        >
          {{ t('recipeEditor.clearSelection') }}
        </NButton>
        <NButton
          size="small"
          quaternary
          :disabled="replacing || generating"
          :style="canUndo ? undefined : { visibility: 'hidden', pointerEvents: 'none' }"
          @click="handleUndo"
        >
          {{ t('recipeEditor.undo') }}
        </NButton>
      </NSpace>
    </template>

    <NAlert v-if="summaryError" type="error" style="margin-bottom: 12px">
      {{ summaryError }}
    </NAlert>

    <div v-if="summary" class="recipe-editor-content">
      <div
        ref="previewAreaRef"
        class="recipe-editor-preview"
        @mousedown="recordMouseDown"
        @click="handleViewportClick"
        @mousemove="handlePreviewMouseMove"
        @mouseleave="handlePreviewMouseLeave"
      >
        <ZoomableImageViewport
          :src="previewBlobUrl"
          alt="recipe preview"
          :height="420"
          :controller="panZoom"
          :content-width="summary?.width ?? 0"
          :content-height="summary?.height ?? 0"
        >
          <template #default="{ transform, effectiveScale }">
            <RegionOverlayCanvas
              :region-map="regionMap"
              :selected-region-ids="selectedRegionIds"
              :transform="transform"
              :effective-scale="effectiveScale"
              :source-width="summary?.width ?? 0"
              :source-height="summary?.height ?? 0"
            />
          </template>
        </ZoomableImageViewport>
        <NButton
          v-if="isViewTransformed"
          class="recipe-editor-reset-zoom"
          size="tiny"
          secondary
          @click.stop="panZoom.resetView()"
        >
          {{ t('recipeEditor.resetZoom') }}
        </NButton>
      </div>
      <div class="recipe-editor-infobar">
        <template v-if="hoverInfo">
          <span>{{ hoverInfo.px }}, {{ hoverInfo.py }}</span>
          <template v-if="hoverInfo.regionId !== null">
            <span class="recipe-editor-infobar__sep">·</span>
            <span>{{ t('recipeEditor.infoRegion', { id: hoverInfo.regionId }) }}</span>
          </template>
          <template v-if="hoverInfo.recipeLabel">
            <span class="recipe-editor-infobar__sep">·</span>
            <span
              class="recipe-editor-infobar__swatch"
              :style="{ backgroundColor: hoverInfo.recipeHex ?? 'transparent' }"
            />
            <span style="font-family: monospace">{{ hoverInfo.recipeLabel }}</span>
          </template>
        </template>
        <span v-else class="recipe-editor-infobar__hint">
          {{ t('recipeEditor.infobarHint') }}
        </span>
      </div>

      <div class="recipe-editor-panels">
        <RecipeSummaryPanel
          :summary="summary"
          :selected-recipe-index="selectedRecipeIndex"
          @select-recipe="handleSelectRecipe"
        />
        <div class="candidate-panel-wrapper">
          <NTabs v-model:value="rightTab" type="line" size="small" class="candidate-tabs">
            <NTabPane
              name="alternatives"
              :tab="t('recipeEditor.tabs.alternatives')"
              class="candidate-tab-pane"
            >
              <RecipeCandidatePanel
                :task-id="taskId"
                :target-lab="targetLab"
                :target-hex="targetHex"
                :palette="summary.palette"
                @select="handleCandidateSelect"
              />
            </NTabPane>
            <NTabPane name="custom" :tab="t('recipeEditor.tabs.custom')" class="candidate-tab-pane">
              <CustomRecipeListPanel
                :items="customRecipes"
                :palette="summary.palette"
                :has-selection="hasSelection"
                @select="handleCandidateSelect"
                @create="openCustomRecipeDialog"
              />
            </NTabPane>
          </NTabs>
        </div>
      </div>

      <CustomRecipeDialog
        v-model:show="showCustomRecipeDialog"
        :task-id="taskId"
        :initial-recipe="customRecipeInitialRecipe"
        :initial-hex="customRecipeInitialHex"
        :palette="summary.palette"
        :current-color-layers="summary.color_layers"
        @confirm="handleCustomRecipeConfirm"
      />
    </div>

    <NText v-if="replacing" depth="3" style="font-size: 12px; margin-top: 8px; display: block">
      {{ t('recipeEditor.replacing') }}
    </NText>

    <NAlert v-if="generateDone && !editedAfterGenerate" type="success" style="margin-top: 8px">
      {{ t('recipeEditor.generateSuccess') }}
    </NAlert>

    <NAlert v-if="editedAfterGenerate" type="warning" style="margin-top: 8px">
      {{ t('recipeEditor.staleWarning') }}
    </NAlert>

    <NAlert
      v-if="generateError"
      type="error"
      style="margin-top: 8px"
      closable
      @close="generateError = null"
    >
      {{ generateError }}
    </NAlert>

    <NSpace justify="end" align="center" style="margin-top: 12px">
      <NButton
        type="primary"
        :loading="generating"
        :disabled="generating || replacing || !summary"
        @click="handleGenerate"
      >
        {{
          generating
            ? t('recipeEditor.generating')
            : editedAfterGenerate
              ? t('recipeEditor.regenerate')
              : t('recipeEditor.generate')
        }}
      </NButton>
      <NButton
        v-if="generateDone"
        type="success"
        :loading="downloading3mf"
        @click="handleDownload3MF"
      >
        {{ t('recipeEditor.download3mf') }}
      </NButton>
    </NSpace>
  </NCard>
</template>

<style scoped>
.recipe-editor-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.recipe-editor-preview {
  position: relative;
  cursor: crosshair;
}

.recipe-editor-preview :deep(.zoomable-image-viewport) {
  cursor: crosshair;
}

.recipe-editor-preview :deep(.zoomable-image-viewport:active) {
  cursor: crosshair;
}

.recipe-editor-reset-zoom {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 10;
}

.recipe-editor-infobar {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  font-size: 12px;
  color: var(--n-text-color-3, #999);
  background: var(--n-color, #fafafa);
  border: 1px solid var(--n-border-color, #e0e0e0);
  border-top: none;
  border-radius: 0 0 4px 4px;
}

.recipe-editor-infobar__sep {
  margin: 0 2px;
  color: var(--n-text-color-disabled, #ccc);
}

.recipe-editor-infobar__swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 2px;
  border: 1px solid rgba(128, 128, 128, 0.3);
  vertical-align: middle;
}

.recipe-editor-infobar__hint {
  font-style: italic;
  color: var(--n-text-color-disabled, #bbb);
}

.recipe-editor-panels {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr;
  gap: 16px;
  height: 420px;
}

.candidate-panel-wrapper {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.candidate-tabs {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.candidate-tabs :deep(.n-tabs-nav) {
  flex-shrink: 0;
}

.candidate-tabs :deep(.n-tab-pane) {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.candidate-tabs :deep(.n-tabs-pane-wrapper) {
  flex: 1;
  min-height: 0;
}

.candidate-tab-pane {
  height: 100%;
}

@media (max-width: 720px) {
  .recipe-editor-panels {
    grid-template-columns: 1fr;
  }
}
</style>
