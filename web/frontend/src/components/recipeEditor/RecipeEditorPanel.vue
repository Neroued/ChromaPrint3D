<script setup lang="ts">
import { computed, ref, watch, onUnmounted, nextTick } from 'vue'
import type { CSSProperties } from 'vue'
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
import { useObjectUrlLifecycle } from '../../composables/useObjectUrlLifecycle'
import { useBlobDownload } from '../../composables/useBlobDownload'
import { getResultPath } from '../../services/resultService'
import { trackEvent, toDurationMs, shortenError, resolveErrorCode } from '../../services/analytics'
import { createSvgPreviewWithoutStroke } from '../../domain/upload/svgPreview'

const props = defineProps<{
  taskId: string
}>()

const { t } = useI18n()
const message = useMessage()
const appStore = useAppStore()
const { downloadByUrl } = useBlobDownload()
const { createUrl: createManagedUrl, revokeUrl: revokeManagedUrl } = useObjectUrlLifecycle()

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
const fullscreenOpen = ref(false)
const originalCompareCollapsed = ref(false)
const originalPreviewUrl = ref('')

interface UndoEntry {
  regionIds: number[]
  oldRecipe: RecipeInfo
  oldRecipeIndex: number
}
const undoStack = ref<UndoEntry[]>([])

const panZoom = usePanZoom()
const previewAreaRef = ref<HTMLElement | null>(null)
const fullscreenPreviewAreaRef = ref<HTMLElement | null>(null)
const fullscreenBodyRef = ref<HTMLElement | null>(null)
const fullscreenSideRef = ref<HTMLElement | null>(null)
const mouseDownPos = ref<{ x: number; y: number } | null>(null)
const hoverInfo = ref<{
  px: number
  py: number
  regionId: number | null
  recipeLabel: string | null
  recipeHex: string | null
} | null>(null)

interface OriginalCompareBox {
  left: number
  top: number
  width: number
  height: number
}

const FULLSCREEN_SIDE_DEFAULT_WIDTH = 420
const FULLSCREEN_SIDE_MIN_WIDTH = 340
const FULLSCREEN_SIDE_MAX_WIDTH = 620
const FULLSCREEN_PREVIEW_MIN_WIDTH = 460
const FULLSCREEN_RESIZER_SIZE = 12
const FULLSCREEN_SUMMARY_DEFAULT_HEIGHT = 260
const FULLSCREEN_SUMMARY_MIN_HEIGHT = 170
const FULLSCREEN_CANDIDATE_MIN_HEIGHT = 230
const ORIGINAL_COMPARE_MARGIN = 14
const ORIGINAL_COMPARE_DEFAULT_WIDTH = 320
const ORIGINAL_COMPARE_DEFAULT_HEIGHT = 230
const ORIGINAL_COMPARE_MIN_WIDTH = 180
const ORIGINAL_COMPARE_MIN_HEIGHT = 140
const ORIGINAL_COMPARE_CHROME_HEIGHT = 48

const fullscreenSideWidth = ref(FULLSCREEN_SIDE_DEFAULT_WIDTH)
const fullscreenSummaryHeight = ref(FULLSCREEN_SUMMARY_DEFAULT_HEIGHT)
const originalCompareBox = ref<OriginalCompareBox>({
  left: ORIGINAL_COMPARE_MARGIN,
  top: ORIGINAL_COMPARE_MARGIN,
  width: ORIGINAL_COMPARE_DEFAULT_WIDTH,
  height: ORIGINAL_COMPARE_DEFAULT_HEIGHT,
})

let originalComparePlacementInitialized = false

const isViewTransformed = computed(
  () =>
    panZoom.scale.value !== 1 || panZoom.translateX.value !== 0 || panZoom.translateY.value !== 0,
)

const fullscreenBodyStyle = computed<CSSProperties>(() => ({
  '--recipe-editor-side-width': `${fullscreenSideWidth.value}px`,
}))

const fullscreenSideStyle = computed<CSSProperties>(() => ({
  '--recipe-editor-summary-height': `${fullscreenSummaryHeight.value}px`,
}))

const originalCompareStyle = computed<CSSProperties>(() => ({
  left: `${originalCompareBox.value.left}px`,
  top: `${originalCompareBox.value.top}px`,
  width: `${originalCompareBox.value.width}px`,
  height: `${originalCompareBox.value.height}px`,
}))

const originalCompareToggleStyle = computed<CSSProperties>(() => ({
  left: `${originalCompareBox.value.left}px`,
  top: `${originalCompareBox.value.top}px`,
}))

const originalCompareViewportHeight = computed(
  () => `${Math.max(90, originalCompareBox.value.height - ORIGINAL_COMPARE_CHROME_HEIGHT)}px`,
)

const generateButtonLabel = computed(() =>
  generating.value
    ? t('recipeEditor.generating')
    : editedAfterGenerate.value
      ? t('recipeEditor.regenerate')
      : t('recipeEditor.generate'),
)

const canGenerate = computed(() => !generating.value && !replacing.value && Boolean(summary.value))

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

function screenToImagePixel(
  clientX: number,
  clientY: number,
  el: HTMLElement | null,
): { x: number; y: number } | null {
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

function handlePreviewMouseMove(event: MouseEvent, el: HTMLElement | null) {
  if (!regionMap.value || !summary.value) {
    hoverInfo.value = null
    return
  }
  const pixel = screenToImagePixel(event.clientX, event.clientY, el)
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

function handleInlinePreviewMouseMove(event: MouseEvent) {
  handlePreviewMouseMove(event, previewAreaRef.value)
}

function handleFullscreenPreviewMouseMove(event: MouseEvent) {
  handlePreviewMouseMove(event, fullscreenPreviewAreaRef.value)
}

// ── Click / drag handling ────────────────────────────────────────────────────

function recordMouseDown(event: MouseEvent) {
  mouseDownPos.value = { x: event.clientX, y: event.clientY }
}

function handleViewportClick(event: MouseEvent, el: HTMLElement | null) {
  if (!regionMap.value || !summary.value) return

  if (mouseDownPos.value) {
    const dx = event.clientX - mouseDownPos.value.x
    const dy = event.clientY - mouseDownPos.value.y
    if (dx * dx + dy * dy > 25) return
  }

  const pixel = screenToImagePixel(event.clientX, event.clientY, el)
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

function handleInlineViewportClick(event: MouseEvent) {
  handleViewportClick(event, previewAreaRef.value)
}

function handleFullscreenViewportClick(event: MouseEvent) {
  handleViewportClick(event, fullscreenPreviewAreaRef.value)
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
    trackEvent('recipe-replace', { region_count: regionIds.length })
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
  trackEvent('recipe-undo')
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
  const startedAt = performance.now()
  try {
    await submitGenerateModel(props.taskId)
    const status = await waitForRecipeGeneration(props.taskId, {
      failedMessage: t('recipeEditor.generateFailed'),
      timeoutMessage: t('recipeEditor.generateTimeout'),
    })
    appStore.setCompletedTask(status)
    generateDone.value = true
    trackEvent('recipe-generate-complete', {
      duration_ms: toDurationMs(performance.now() - startedAt),
    })
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    generateError.value = msg
    trackEvent('recipe-generate-fail', {
      error: shortenError(e),
      error_code: resolveErrorCode(e),
    })
  } finally {
    generating.value = false
  }
}

async function handleDownload3MF() {
  if (downloading3mf.value) return
  downloading3mf.value = true
  const baseName = appStore.selectedFile?.name?.replace(/\.[^.]+$/, '') ?? 'result'
  const filename = `${baseName}.3mf`
  try {
    await downloadByUrl(getResultPath(props.taskId), filename)
    trackEvent('recipe-download-3mf', { filename })
  } catch {
    /* handled by runtime */
  } finally {
    downloading3mf.value = false
  }
}

// ── Fullscreen workspace ─────────────────────────────────────────────────────

let bodyOverflowBeforeFullscreen: string | null = null
let originalPreviewRequestId = 0

type FullscreenDragInteraction =
  | {
      type: 'original-drag'
      startX: number
      startY: number
      startLeft: number
      startTop: number
    }
  | {
      type: 'original-resize'
      startX: number
      startY: number
      startWidth: number
      startHeight: number
    }
  | {
      type: 'side-resize'
      startX: number
      startWidth: number
    }
  | {
      type: 'summary-resize'
      startY: number
      startHeight: number
    }

let activeFullscreenDrag: FullscreenDragInteraction | null = null

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max))
}

function getFullscreenPreviewSize(): { width: number; height: number } {
  const el = fullscreenPreviewAreaRef.value
  return {
    width: el?.clientWidth ?? 0,
    height: el?.clientHeight ?? 0,
  }
}

function clampOriginalCompareBox(box: OriginalCompareBox): OriginalCompareBox {
  const bounds = getFullscreenPreviewSize()
  const maxWidth =
    bounds.width > 0
      ? Math.max(ORIGINAL_COMPARE_MIN_WIDTH, bounds.width - ORIGINAL_COMPARE_MARGIN * 2)
      : ORIGINAL_COMPARE_DEFAULT_WIDTH
  const maxHeight =
    bounds.height > 0
      ? Math.max(ORIGINAL_COMPARE_MIN_HEIGHT, bounds.height - ORIGINAL_COMPARE_MARGIN * 2)
      : ORIGINAL_COMPARE_DEFAULT_HEIGHT
  const width = clampNumber(box.width, ORIGINAL_COMPARE_MIN_WIDTH, maxWidth)
  const height = clampNumber(box.height, ORIGINAL_COMPARE_MIN_HEIGHT, maxHeight)
  const maxLeft =
    bounds.width > 0
      ? Math.max(ORIGINAL_COMPARE_MARGIN, bounds.width - width - ORIGINAL_COMPARE_MARGIN)
      : ORIGINAL_COMPARE_MARGIN
  const maxTop =
    bounds.height > 0
      ? Math.max(ORIGINAL_COMPARE_MARGIN, bounds.height - height - ORIGINAL_COMPARE_MARGIN)
      : ORIGINAL_COMPARE_MARGIN

  return {
    left: clampNumber(box.left, ORIGINAL_COMPARE_MARGIN, maxLeft),
    top: clampNumber(box.top, ORIGINAL_COMPARE_MARGIN, maxTop),
    width,
    height,
  }
}

function setOriginalCompareBox(box: OriginalCompareBox) {
  originalCompareBox.value = clampOriginalCompareBox(box)
}

function resetOriginalComparePlacement() {
  const bounds = getFullscreenPreviewSize()
  const width =
    bounds.width > 0
      ? Math.min(ORIGINAL_COMPARE_DEFAULT_WIDTH, bounds.width - ORIGINAL_COMPARE_MARGIN * 2)
      : ORIGINAL_COMPARE_DEFAULT_WIDTH
  const height =
    bounds.height > 0
      ? Math.min(ORIGINAL_COMPARE_DEFAULT_HEIGHT, bounds.height - ORIGINAL_COMPARE_MARGIN * 2)
      : ORIGINAL_COMPARE_DEFAULT_HEIGHT

  setOriginalCompareBox({
    left: ORIGINAL_COMPARE_MARGIN,
    top: ORIGINAL_COMPARE_MARGIN,
    width,
    height,
  })
  originalComparePlacementInitialized = true
}

function clampFullscreenSideWidth(width = fullscreenSideWidth.value): number {
  const bodyWidth = fullscreenBodyRef.value?.clientWidth ?? 0
  const maxWidth =
    bodyWidth > 0
      ? Math.min(
          FULLSCREEN_SIDE_MAX_WIDTH,
          bodyWidth - FULLSCREEN_PREVIEW_MIN_WIDTH - FULLSCREEN_RESIZER_SIZE,
        )
      : FULLSCREEN_SIDE_MAX_WIDTH
  return clampNumber(width, FULLSCREEN_SIDE_MIN_WIDTH, maxWidth)
}

function clampFullscreenSummaryHeight(height = fullscreenSummaryHeight.value): number {
  const side = fullscreenSideRef.value
  if (!side) {
    return clampNumber(height, FULLSCREEN_SUMMARY_MIN_HEIGHT, FULLSCREEN_SUMMARY_DEFAULT_HEIGHT)
  }
  const status = side.querySelector<HTMLElement>('.recipe-editor-fullscreen__status')
  const statusHeight = status ? status.offsetHeight + 12 : 0
  const maxHeight =
    side.clientHeight - statusHeight - FULLSCREEN_CANDIDATE_MIN_HEIGHT - FULLSCREEN_RESIZER_SIZE
  return clampNumber(height, FULLSCREEN_SUMMARY_MIN_HEIGHT, maxHeight)
}

function clampFullscreenLayout() {
  fullscreenSideWidth.value = clampFullscreenSideWidth()
  fullscreenSummaryHeight.value = clampFullscreenSummaryHeight()
  originalCompareBox.value = clampOriginalCompareBox(originalCompareBox.value)
}

function handleFullscreenDragMove(event: MouseEvent) {
  if (!activeFullscreenDrag) return

  if (activeFullscreenDrag.type === 'original-drag') {
    const dx = event.clientX - activeFullscreenDrag.startX
    const dy = event.clientY - activeFullscreenDrag.startY
    setOriginalCompareBox({
      ...originalCompareBox.value,
      left: activeFullscreenDrag.startLeft + dx,
      top: activeFullscreenDrag.startTop + dy,
    })
    return
  }

  if (activeFullscreenDrag.type === 'original-resize') {
    const dx = event.clientX - activeFullscreenDrag.startX
    const dy = event.clientY - activeFullscreenDrag.startY
    setOriginalCompareBox({
      ...originalCompareBox.value,
      width: activeFullscreenDrag.startWidth + dx,
      height: activeFullscreenDrag.startHeight + dy,
    })
    return
  }

  if (activeFullscreenDrag.type === 'side-resize') {
    const dx = event.clientX - activeFullscreenDrag.startX
    fullscreenSideWidth.value = clampFullscreenSideWidth(activeFullscreenDrag.startWidth - dx)
    return
  }

  const dy = event.clientY - activeFullscreenDrag.startY
  fullscreenSummaryHeight.value = clampFullscreenSummaryHeight(
    activeFullscreenDrag.startHeight + dy,
  )
}

function stopFullscreenDrag() {
  if (typeof window !== 'undefined') {
    window.removeEventListener('mousemove', handleFullscreenDragMove, true)
    window.removeEventListener('mouseup', stopFullscreenDrag, true)
  }
  activeFullscreenDrag = null
  clampFullscreenLayout()
}

function startFullscreenDrag(interaction: FullscreenDragInteraction, event: MouseEvent) {
  event.preventDefault()
  activeFullscreenDrag = interaction
  if (typeof window === 'undefined') return
  window.addEventListener('mousemove', handleFullscreenDragMove, true)
  window.addEventListener('mouseup', stopFullscreenDrag, true)
}

function startOriginalCompareDrag(event: MouseEvent) {
  startFullscreenDrag(
    {
      type: 'original-drag',
      startX: event.clientX,
      startY: event.clientY,
      startLeft: originalCompareBox.value.left,
      startTop: originalCompareBox.value.top,
    },
    event,
  )
}

function startOriginalCompareResize(event: MouseEvent) {
  startFullscreenDrag(
    {
      type: 'original-resize',
      startX: event.clientX,
      startY: event.clientY,
      startWidth: originalCompareBox.value.width,
      startHeight: originalCompareBox.value.height,
    },
    event,
  )
}

function startFullscreenSideResize(event: MouseEvent) {
  startFullscreenDrag(
    {
      type: 'side-resize',
      startX: event.clientX,
      startWidth: fullscreenSideWidth.value,
    },
    event,
  )
}

function startFullscreenSummaryResize(event: MouseEvent) {
  startFullscreenDrag(
    {
      type: 'summary-resize',
      startY: event.clientY,
      startHeight: fullscreenSummaryHeight.value,
    },
    event,
  )
}

function handleFullscreenWindowResize() {
  if (!fullscreenOpen.value) return
  clampFullscreenLayout()
}

function isSvgFile(file: File): boolean {
  return file.type === 'image/svg+xml' || file.name.toLowerCase().endsWith('.svg')
}

function restoreBodyScroll() {
  if (typeof document === 'undefined' || bodyOverflowBeforeFullscreen === null) return
  document.body.style.overflow = bodyOverflowBeforeFullscreen
  bodyOverflowBeforeFullscreen = null
}

function openFullscreenEditor() {
  if (!summary.value) return
  fullscreenOpen.value = true
  originalCompareCollapsed.value = false
  panZoom.resetView()
  void nextTick(() => {
    panZoom.resetView()
    if (originalComparePlacementInitialized) {
      clampFullscreenLayout()
      return
    }
    resetOriginalComparePlacement()
    clampFullscreenLayout()
  })
}

function closeFullscreenEditor() {
  fullscreenOpen.value = false
}

async function refreshOriginalPreview(file: File | null) {
  originalPreviewRequestId += 1
  const requestId = originalPreviewRequestId
  revokeManagedUrl(originalPreviewUrl.value)
  originalPreviewUrl.value = ''
  if (!file) return

  const previewBlob = isSvgFile(file)
    ? await createSvgPreviewWithoutStroke(file).catch(() => file)
    : file

  if (requestId !== originalPreviewRequestId || appStore.selectedFile !== file) return
  originalPreviewUrl.value = createManagedUrl(previewBlob)
}

watch(fullscreenOpen, (open) => {
  if (typeof document === 'undefined') return
  if (open) {
    if (bodyOverflowBeforeFullscreen === null) {
      bodyOverflowBeforeFullscreen = document.body.style.overflow
    }
    document.body.style.overflow = 'hidden'
    if (typeof window !== 'undefined') {
      window.addEventListener('resize', handleFullscreenWindowResize)
    }
    return
  }
  stopFullscreenDrag()
  if (typeof window !== 'undefined') {
    window.removeEventListener('resize', handleFullscreenWindowResize)
  }
  restoreBodyScroll()
})

watch(
  () => appStore.selectedFile,
  (file) => {
    void refreshOriginalPreview(file)
  },
  { immediate: true },
)

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
    fullscreenOpen.value = false
    originalCompareCollapsed.value = false
    originalComparePlacementInitialized = false
    stopKeepalive()
    clearRegionMap()
    panZoom.resetView()
    if (props.taskId) void loadSummary()
  },
  { immediate: true },
)

watch(globalMode, (enabled) => {
  trackEvent('recipe-global-mode-toggle', { enabled })
})

onUnmounted(() => {
  stopKeepalive()
  stopFullscreenDrag()
  if (typeof window !== 'undefined') {
    window.removeEventListener('resize', handleFullscreenWindowResize)
  }
  restoreBodyScroll()
  originalPreviewRequestId += 1
  revokeManagedUrl(originalPreviewUrl.value)
  originalPreviewUrl.value = ''
})
</script>

<template>
  <NCard :title="t('recipeEditor.title')" size="small">
    <template #header-extra>
      <NSpace :size="12" align="center">
        <NButton size="small" secondary :disabled="!summary" @click="openFullscreenEditor">
          {{ t('recipeEditor.fullscreenOpen') }}
        </NButton>
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
        @click="handleInlineViewportClick"
        @mousemove="handleInlinePreviewMouseMove"
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
        v-if="!fullscreenOpen"
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
        {{ generateButtonLabel }}
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

  <Teleport to="body">
    <template v-if="fullscreenOpen && summary">
      <div class="recipe-editor-fullscreen">
        <div class="recipe-editor-fullscreen__toolbar">
          <NSpace align="center" :size="12" class="recipe-editor-fullscreen__title-group">
            <NText strong class="recipe-editor-fullscreen__title">
              {{ t('recipeEditor.fullscreenTitle') }}
            </NText>
            <NText depth="3" class="recipe-editor-fullscreen__meta">
              {{ summary.width }} × {{ summary.height }}
            </NText>
          </NSpace>

          <NSpace align="center" :size="10" class="recipe-editor-fullscreen__actions">
            <NTooltip>
              <template #trigger>
                <NSpace :size="6" align="center">
                  <NText depth="3" class="recipe-editor-fullscreen__tool-label">
                    {{ t('recipeEditor.globalModeLabel') }}
                  </NText>
                  <NSwitch v-model:value="globalMode" size="small" />
                </NSpace>
              </template>
              {{ t('recipeEditor.globalModeTooltip') }}
            </NTooltip>
            <NButton size="small" secondary :disabled="!hasSelection" @click="clearSelection">
              {{ t('recipeEditor.clearSelection') }}
            </NButton>
            <NButton
              size="small"
              secondary
              :disabled="!canUndo || replacing || generating"
              @click="handleUndo"
            >
              {{ t('recipeEditor.undo') }}
            </NButton>
            <NButton
              type="primary"
              size="small"
              :loading="generating"
              :disabled="!canGenerate"
              @click="handleGenerate"
            >
              {{ generateButtonLabel }}
            </NButton>
            <NButton
              v-if="generateDone"
              type="success"
              size="small"
              :loading="downloading3mf"
              @click="handleDownload3MF"
            >
              {{ t('recipeEditor.download3mf') }}
            </NButton>
            <NButton size="small" quaternary @click="closeFullscreenEditor">
              {{ t('recipeEditor.fullscreenClose') }}
            </NButton>
          </NSpace>
        </div>

        <div
          ref="fullscreenBodyRef"
          class="recipe-editor-fullscreen__body"
          :style="fullscreenBodyStyle"
        >
          <section class="recipe-editor-fullscreen__stage">
            <div
              ref="fullscreenPreviewAreaRef"
              class="recipe-editor-preview recipe-editor-preview--fullscreen"
              @mousedown="recordMouseDown"
              @click="handleFullscreenViewportClick"
              @mousemove="handleFullscreenPreviewMouseMove"
              @mouseleave="handlePreviewMouseLeave"
            >
              <ZoomableImageViewport
                :src="previewBlobUrl"
                alt="recipe preview"
                height="100%"
                :controller="panZoom"
                :content-width="summary.width"
                :content-height="summary.height"
              >
                <template #default="{ transform, effectiveScale }">
                  <RegionOverlayCanvas
                    :region-map="regionMap"
                    :selected-region-ids="selectedRegionIds"
                    :transform="transform"
                    :effective-scale="effectiveScale"
                    :source-width="summary.width"
                    :source-height="summary.height"
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

              <div
                v-if="originalPreviewUrl && !originalCompareCollapsed"
                class="recipe-editor-original-compare"
                :style="originalCompareStyle"
                @mousedown.stop
                @mousemove.stop
                @click.stop
                @wheel.stop
              >
                <div
                  class="recipe-editor-original-compare__header"
                  @mousedown.stop.prevent="startOriginalCompareDrag"
                >
                  <NText strong class="recipe-editor-original-compare__title">
                    {{ t('recipeEditor.originalCompare') }}
                  </NText>
                  <NButton
                    size="tiny"
                    quaternary
                    @mousedown.stop
                    @click="originalCompareCollapsed = true"
                  >
                    {{ t('recipeEditor.originalCompareCollapse') }}
                  </NButton>
                </div>
                <ZoomableImageViewport
                  :src="originalPreviewUrl"
                  alt="original preview"
                  :height="originalCompareViewportHeight"
                  :checkerboard="true"
                />
                <div
                  class="recipe-editor-original-compare__resize-handle"
                  role="separator"
                  :aria-label="t('recipeEditor.resizeOriginalCompare')"
                  @mousedown.stop.prevent="startOriginalCompareResize"
                />
              </div>

              <NButton
                v-else-if="originalPreviewUrl"
                class="recipe-editor-original-compare-toggle"
                :style="originalCompareToggleStyle"
                size="tiny"
                secondary
                @mousedown.stop
                @click.stop="originalCompareCollapsed = false"
              >
                {{ t('recipeEditor.originalCompareExpand') }}
              </NButton>
            </div>

            <div class="recipe-editor-infobar recipe-editor-infobar--fullscreen">
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
          </section>

          <div
            class="recipe-editor-fullscreen__vertical-resizer"
            role="separator"
            aria-orientation="vertical"
            :aria-label="t('recipeEditor.resizeEditorColumns')"
            @mousedown="startFullscreenSideResize"
          />

          <aside
            ref="fullscreenSideRef"
            class="recipe-editor-fullscreen__side"
            :style="fullscreenSideStyle"
          >
            <div class="recipe-editor-fullscreen__status">
              <NText v-if="replacing" depth="3" class="recipe-editor-fullscreen__status-text">
                {{ t('recipeEditor.replacing') }}
              </NText>
              <NAlert v-if="generateDone && !editedAfterGenerate" type="success" :bordered="false">
                {{ t('recipeEditor.generateSuccess') }}
              </NAlert>
              <NAlert v-if="editedAfterGenerate" type="warning" :bordered="false">
                {{ t('recipeEditor.staleWarning') }}
              </NAlert>
              <NAlert
                v-if="generateError"
                type="error"
                closable
                :bordered="false"
                @close="generateError = null"
              >
                {{ generateError }}
              </NAlert>
            </div>

            <div class="recipe-editor-fullscreen__summary-panel">
              <RecipeSummaryPanel
                :summary="summary"
                :selected-recipe-index="selectedRecipeIndex"
                @select-recipe="handleSelectRecipe"
              />
            </div>

            <div
              class="recipe-editor-fullscreen__horizontal-resizer"
              role="separator"
              aria-orientation="horizontal"
              :aria-label="t('recipeEditor.resizeEditorRows')"
              @mousedown="startFullscreenSummaryResize"
            />

            <div class="recipe-editor-fullscreen__candidate-panel">
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
                <NTabPane
                  name="custom"
                  :tab="t('recipeEditor.tabs.custom')"
                  class="candidate-tab-pane"
                >
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
          </aside>
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
    </template>
  </Teleport>
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

.recipe-editor-preview--fullscreen {
  flex: 1;
  min-height: 0;
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

.recipe-editor-fullscreen {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  background: #f4f7fb;
  color: #1f2933;
}

:global([data-theme='dark']) .recipe-editor-fullscreen {
  background: #101722;
  color: #e6edf5;
}

.recipe-editor-fullscreen__toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 56px;
  padding: 10px 18px;
  background: rgba(255, 255, 255, 0.94);
  border-bottom: 1px solid rgba(135, 150, 170, 0.32);
  box-shadow: 0 8px 24px rgba(25, 39, 52, 0.08);
}

:global([data-theme='dark']) .recipe-editor-fullscreen__toolbar {
  background: rgba(22, 30, 42, 0.96);
  border-bottom-color: rgba(128, 148, 170, 0.24);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
}

.recipe-editor-fullscreen__title-group {
  min-width: 0;
}

.recipe-editor-fullscreen__title {
  font-size: 16px;
  white-space: nowrap;
}

.recipe-editor-fullscreen__meta,
.recipe-editor-fullscreen__tool-label,
.recipe-editor-fullscreen__status-text {
  font-size: 12px;
}

.recipe-editor-fullscreen__actions {
  flex-wrap: wrap;
  justify-content: flex-end;
}

.recipe-editor-fullscreen__body {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 12px var(--recipe-editor-side-width, 420px);
  gap: 0;
  padding: 14px;
  overflow: hidden;
}

.recipe-editor-fullscreen__stage {
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  padding-right: 8px;
}

.recipe-editor-fullscreen__stage .recipe-editor-preview {
  min-height: 0;
}

.recipe-editor-fullscreen__stage :deep(.zoomable-image-viewport) {
  background-color: rgba(255, 255, 255, 0.64);
}

:global([data-theme='dark']) .recipe-editor-fullscreen__stage :deep(.zoomable-image-viewport) {
  background-color: rgba(8, 13, 20, 0.62);
}

.recipe-editor-infobar--fullscreen {
  min-height: 28px;
  flex-shrink: 0;
  background: rgba(255, 255, 255, 0.92);
}

:global([data-theme='dark']) .recipe-editor-infobar--fullscreen {
  background: rgba(22, 30, 42, 0.92);
}

.recipe-editor-fullscreen__side {
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows:
    auto minmax(170px, var(--recipe-editor-summary-height, 260px)) 12px
    minmax(230px, 1fr);
  gap: 0;
  padding-left: 8px;
  overflow: hidden;
}

.recipe-editor-fullscreen__status {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
  margin-bottom: 12px;
}

.recipe-editor-fullscreen__summary-panel,
.recipe-editor-fullscreen__candidate-panel {
  min-height: 0;
  overflow: hidden;
}

.recipe-editor-fullscreen__candidate-panel {
  display: flex;
  flex-direction: column;
}

.recipe-editor-fullscreen__vertical-resizer,
.recipe-editor-fullscreen__horizontal-resizer {
  position: relative;
  flex-shrink: 0;
  border-radius: 4px;
}

.recipe-editor-fullscreen__vertical-resizer {
  min-height: 0;
  cursor: col-resize;
}

.recipe-editor-fullscreen__horizontal-resizer {
  min-width: 0;
  cursor: row-resize;
}

.recipe-editor-fullscreen__vertical-resizer::before,
.recipe-editor-fullscreen__horizontal-resizer::before {
  content: '';
  position: absolute;
  border-radius: 999px;
  background: rgba(118, 138, 161, 0.42);
  transition:
    background-color 0.15s ease,
    box-shadow 0.15s ease;
}

.recipe-editor-fullscreen__vertical-resizer::before {
  top: 12px;
  bottom: 12px;
  left: 50%;
  width: 2px;
  transform: translateX(-50%);
}

.recipe-editor-fullscreen__horizontal-resizer::before {
  top: 50%;
  right: 12px;
  left: 12px;
  height: 2px;
  transform: translateY(-50%);
}

.recipe-editor-fullscreen__vertical-resizer:hover::before,
.recipe-editor-fullscreen__horizontal-resizer:hover::before {
  background: #4c8bf5;
  box-shadow: 0 0 0 3px rgba(76, 139, 245, 0.14);
}

.recipe-editor-original-compare {
  position: absolute;
  z-index: 20;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  min-width: 180px;
  min-height: 140px;
  padding: 8px;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(120, 136, 156, 0.35);
  border-radius: 6px;
  box-shadow: 0 10px 28px rgba(17, 31, 44, 0.18);
}

:global([data-theme='dark']) .recipe-editor-original-compare {
  background: rgba(24, 33, 46, 0.94);
  border-color: rgba(128, 148, 170, 0.32);
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.38);
}

.recipe-editor-original-compare__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
  cursor: move;
  user-select: none;
}

.recipe-editor-original-compare__title {
  min-width: 0;
  font-size: 12px;
  white-space: nowrap;
}

.recipe-editor-original-compare__resize-handle {
  position: absolute;
  right: 1px;
  bottom: 1px;
  width: 18px;
  height: 18px;
  cursor: nwse-resize;
}

.recipe-editor-original-compare__resize-handle::before,
.recipe-editor-original-compare__resize-handle::after {
  content: '';
  position: absolute;
  right: 4px;
  bottom: 4px;
  border-right: 1px solid rgba(91, 108, 130, 0.58);
  border-bottom: 1px solid rgba(91, 108, 130, 0.58);
}

.recipe-editor-original-compare__resize-handle::before {
  width: 10px;
  height: 10px;
}

.recipe-editor-original-compare__resize-handle::after {
  width: 5px;
  height: 5px;
}

.recipe-editor-original-compare-toggle {
  position: absolute;
  z-index: 20;
}

@media (max-width: 720px) {
  .recipe-editor-panels {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 900px) {
  .recipe-editor-fullscreen__toolbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .recipe-editor-fullscreen__actions {
    justify-content: flex-start;
  }

  .recipe-editor-fullscreen__body {
    grid-template-columns: 1fr !important;
    grid-template-rows: minmax(420px, 1fr) minmax(360px, 42vh);
    overflow: auto;
  }

  .recipe-editor-fullscreen__stage {
    padding-right: 0;
  }

  .recipe-editor-fullscreen__vertical-resizer,
  .recipe-editor-fullscreen__horizontal-resizer {
    display: none;
  }

  .recipe-editor-fullscreen__side {
    grid-template-rows: auto minmax(170px, 1fr) minmax(220px, 1fr);
    padding-left: 0;
  }
}
</style>
