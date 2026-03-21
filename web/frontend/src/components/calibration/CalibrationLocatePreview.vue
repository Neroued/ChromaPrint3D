<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, computed } from 'vue'
import { NAlert } from 'naive-ui'
import { computeSamplePositions } from '../../utils/homography'
import { useI18n } from 'vue-i18n'
import type { BoardGeometryInfo } from '../../composables/useColorDBBuildFlow'

const { t } = useI18n()

const props = withDefaults(
  defineProps<{
    imageFile: File
    corners?: [number, number][]
    boardGeometry?: BoardGeometryInfo | null
  }>(),
  {
    corners: () => [],
    boardGeometry: null,
  },
)

const emit = defineEmits<{
  'update:corners': [corners: [number, number][]]
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const containerRef = ref<HTMLDivElement | null>(null)
const img = ref<HTMLImageElement | null>(null)
const imgLoaded = ref(false)
const draggingIdx = ref(-1)
const displayScale = ref(1)
const naturalWidth = ref(0)
const naturalHeight = ref(0)

const HANDLE_OUTER = 8
const HANDLE_DOT = 2
const CROSS_LEN = 12
const SAMPLE_R = 3

const CORNER_COLORS: string[] = ['#FF4444', '#44AAFF', '#44FF44', '#FFAA44']

const hasOverlay = computed(() => props.corners.length === 4 && props.boardGeometry != null)

const cornerLabels = computed(() => [
  t('colordb.locate.cornerTL'),
  t('colordb.locate.cornerTR'),
  t('colordb.locate.cornerBR'),
  t('colordb.locate.cornerBL'),
])

const samplePositions = computed(() => {
  if (!hasOverlay.value || !props.boardGeometry) return []
  const g = props.boardGeometry
  return computeSamplePositions(props.corners, g.gridRows, g.gridCols, {
    boardW: g.boardW,
    boardH: g.boardH,
    margin: g.margin,
    tile: g.tile,
    gap: g.gap,
    offset: g.offset,
  })
})

const isQuadValid = computed(() => {
  if (props.corners.length !== 4) return true
  const pts = props.corners
  let pos = 0
  let neg = 0
  for (let i = 0; i < 4; i++) {
    const a = pts[i]!
    const b = pts[(i + 1) % 4]!
    const c = pts[(i + 2) % 4]!
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    if (cross > 0) pos++
    else if (cross < 0) neg++
  }
  return pos === 0 || neg === 0
})

function loadImage() {
  const image = new Image()
  image.onload = () => {
    img.value = image
    naturalWidth.value = image.naturalWidth
    naturalHeight.value = image.naturalHeight
    imgLoaded.value = true
    updateDisplayScale()
    draw()
  }
  image.src = URL.createObjectURL(props.imageFile)
}

function updateDisplayScale() {
  if (!containerRef.value || naturalWidth.value === 0) return
  const containerWidth = containerRef.value.clientWidth
  const maxWidth = Math.min(containerWidth, 900)
  displayScale.value = Math.min(1, maxWidth / naturalWidth.value)
}

function drawHandle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  label: string,
) {
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.arc(x, y, HANDLE_OUTER, 0, Math.PI * 2)
  ctx.stroke()

  ctx.beginPath()
  ctx.moveTo(x - CROSS_LEN, y)
  ctx.lineTo(x - HANDLE_OUTER - 2, y)
  ctx.moveTo(x + HANDLE_OUTER + 2, y)
  ctx.lineTo(x + CROSS_LEN, y)
  ctx.moveTo(x, y - CROSS_LEN)
  ctx.lineTo(x, y - HANDLE_OUTER - 2)
  ctx.moveTo(x, y + HANDLE_OUTER + 2)
  ctx.lineTo(x, y + CROSS_LEN)
  ctx.stroke()

  ctx.fillStyle = color
  ctx.beginPath()
  ctx.arc(x, y, HANDLE_DOT, 0, Math.PI * 2)
  ctx.fill()

  ctx.font = 'bold 11px sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'bottom'
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)'
  ctx.lineWidth = 3
  ctx.strokeText(label, x + HANDLE_OUTER + 3, y - 3)
  ctx.fillStyle = color
  ctx.fillText(label, x + HANDLE_OUTER + 3, y - 3)
}

function draw() {
  const canvas = canvasRef.value
  const image = img.value
  if (!canvas || !image || naturalWidth.value === 0) return

  const s = displayScale.value
  const w = Math.round(naturalWidth.value * s)
  const h = Math.round(naturalHeight.value * s)
  canvas.width = w
  canvas.height = h
  canvas.style.width = `${w}px`
  canvas.style.height = `${h}px`

  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.drawImage(image, 0, 0, w, h)

  if (!hasOverlay.value) return

  for (const sp of samplePositions.value) {
    ctx.fillStyle = 'rgba(0, 200, 100, 0.7)'
    ctx.beginPath()
    ctx.arc(sp.x * s, sp.y * s, SAMPLE_R, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.strokeStyle = isQuadValid.value ? 'rgba(255, 80, 80, 0.6)' : 'rgba(255, 0, 0, 1)'
  ctx.lineWidth = isQuadValid.value ? 1.5 : 2.5
  ctx.setLineDash(isQuadValid.value ? [] : [6, 4])
  ctx.beginPath()
  for (let i = 0; i < 4; i++) {
    const corner = props.corners[i]!
    const next = props.corners[(i + 1) % 4]!
    ctx.moveTo(corner[0] * s, corner[1] * s)
    ctx.lineTo(next[0] * s, next[1] * s)
  }
  ctx.stroke()
  ctx.setLineDash([])

  for (let i = 0; i < props.corners.length; i++) {
    const [cx, cy] = props.corners[i]!
    drawHandle(ctx, cx * s, cy * s, CORNER_COLORS[i]!, String(i + 1))
  }
}

function getCanvasPos(e: MouseEvent): [number, number] {
  const canvas = canvasRef.value
  if (!canvas) return [0, 0]
  const rect = canvas.getBoundingClientRect()
  return [e.clientX - rect.left, e.clientY - rect.top]
}

function onPointerDown(e: MouseEvent) {
  if (!hasOverlay.value) return
  const [mx, my] = getCanvasPos(e)
  const s = displayScale.value
  const hitR2 = CROSS_LEN * CROSS_LEN
  for (let i = 0; i < props.corners.length; i++) {
    const corner = props.corners[i]!
    const dx = corner[0] * s - mx
    const dy = corner[1] * s - my
    if (dx * dx + dy * dy <= hitR2) {
      draggingIdx.value = i
      e.preventDefault()
      return
    }
  }
}

function onPointerMove(e: MouseEvent) {
  if (draggingIdx.value < 0) return
  const [mx, my] = getCanvasPos(e)
  const s = displayScale.value
  const updated = props.corners.map((c) => [...c] as [number, number])
  updated[draggingIdx.value] = [
    Math.max(0, Math.min(naturalWidth.value, mx / s)),
    Math.max(0, Math.min(naturalHeight.value, my / s)),
  ]
  emit('update:corners', updated)
}

function onPointerUp() {
  draggingIdx.value = -1
}

watch(
  () => props.corners,
  () => draw(),
  { deep: true },
)

watch(
  () => props.boardGeometry,
  () => draw(),
)

watch(
  () => props.imageFile,
  () => {
    imgLoaded.value = false
    if (img.value?.src) URL.revokeObjectURL(img.value.src)
    loadImage()
  },
)

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  loadImage()
  if (containerRef.value) {
    resizeObserver = new ResizeObserver(() => {
      updateDisplayScale()
      draw()
    })
    resizeObserver.observe(containerRef.value)
  }
})

onUnmounted(() => {
  resizeObserver?.disconnect()
  if (img.value?.src) URL.revokeObjectURL(img.value.src)
})
</script>

<template>
  <div ref="containerRef" class="locate-preview">
    <div v-if="hasOverlay" class="locate-preview__legend">
      <span v-for="(label, i) in cornerLabels" :key="i" class="locate-preview__legend-item">
        <span class="locate-preview__legend-dot" :style="{ backgroundColor: CORNER_COLORS[i] }" />
        {{ label }}
      </span>
      <span class="locate-preview__legend-item">
        <span class="locate-preview__legend-dot" style="background-color: rgba(0, 200, 100, 0.7)" />
        {{ t('colordb.locate.samplePoints') }}
      </span>
    </div>

    <NAlert
      v-if="hasOverlay && !isQuadValid"
      type="warning"
      :bordered="false"
      style="margin-bottom: 8px"
    >
      {{ t('colordb.locate.quadWarning') }}
    </NAlert>

    <canvas
      ref="canvasRef"
      class="locate-preview__canvas"
      :class="{ 'locate-preview__canvas--interactive': hasOverlay }"
      @mousedown="onPointerDown"
      @mousemove="onPointerMove"
      @mouseup="onPointerUp"
      @mouseleave="onPointerUp"
    />

    <p v-if="hasOverlay" class="locate-preview__hint">{{ t('colordb.locate.dragHint') }}</p>
  </div>
</template>

<style scoped>
.locate-preview {
  width: 100%;
}

.locate-preview__canvas {
  display: block;
  border: 1px solid rgba(128, 128, 128, 0.3);
  border-radius: 6px;
}

.locate-preview__canvas--interactive {
  cursor: crosshair;
}

.locate-preview__legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 8px;
  font-size: 12px;
}

.locate-preview__legend-item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.locate-preview__legend-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.6);
}

.locate-preview__hint {
  margin-top: 6px;
  font-size: 12px;
  opacity: 0.65;
}
</style>
