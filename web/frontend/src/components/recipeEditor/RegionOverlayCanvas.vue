<script setup lang="ts">
import { ref, watch } from 'vue'
import type { RegionMapData } from '../../composables/useRegionMap'

const props = defineProps<{
  regionMap: RegionMapData | null
  selectedRegionIds: Set<number>
  transform: string
  effectiveScale: number
  sourceWidth: number
  sourceHeight: number
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const MASK_ALPHA = 102
const BORDER_SCREEN_PX = 1.5
const BORDER_COLOR = 'rgba(255, 255, 255, 0.85)'
const MAX_BORDER_CANVAS_PX = 8

let cachedMaskCanvas: HTMLCanvasElement | null = null
let cachedEdges: Float64Array | null = null
let cachedEdgeCount = 0

function isPixelSelected(regionIds: Uint32Array, idx: number, selected: Set<number>): boolean {
  const rid = regionIds[idx]
  return rid !== undefined && rid !== 0xffffffff && selected.has(rid)
}

function buildMaskAndEdges() {
  const map = props.regionMap
  if (!map || props.selectedRegionIds.size === 0) {
    cachedMaskCanvas = null
    cachedEdges = null
    cachedEdgeCount = 0
    return
  }

  const w = map.width
  const h = map.height
  const regionIds = map.regionIds
  const selected = props.selectedRegionIds

  const tmpCanvas = document.createElement('canvas')
  tmpCanvas.width = w
  tmpCanvas.height = h
  const tmpCtx = tmpCanvas.getContext('2d')!
  const imageData = tmpCtx.createImageData(w, h)
  const data = imageData.data

  for (let i = 0; i < regionIds.length; i++) {
    if (isPixelSelected(regionIds, i, selected)) continue
    data[i * 4 + 3] = MASK_ALPHA
  }
  tmpCtx.putImageData(imageData, 0, 0)
  cachedMaskCanvas = tmpCanvas

  const sw = props.sourceWidth
  const sh = props.sourceHeight
  const scaleX = sw / w
  const scaleY = sh / h

  const edges: number[] = []

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      const sel = isPixelSelected(regionIds, i, selected)

      if (sel) {
        if (x === 0) edges.push(0, y * scaleY, 0, (y + 1) * scaleY)
        if (x === w - 1) edges.push(w * scaleX, y * scaleY, w * scaleX, (y + 1) * scaleY)
        if (y === 0) edges.push(x * scaleX, 0, (x + 1) * scaleX, 0)
        if (y === h - 1) edges.push(x * scaleX, h * scaleY, (x + 1) * scaleX, h * scaleY)
      }

      if (x < w - 1 && sel !== isPixelSelected(regionIds, i + 1, selected)) {
        edges.push((x + 1) * scaleX, y * scaleY, (x + 1) * scaleX, (y + 1) * scaleY)
      }

      if (y < h - 1 && sel !== isPixelSelected(regionIds, i + w, selected)) {
        edges.push(x * scaleX, (y + 1) * scaleY, (x + 1) * scaleX, (y + 1) * scaleY)
      }
    }
  }

  cachedEdges = new Float64Array(edges)
  cachedEdgeCount = edges.length / 4
}

function redraw() {
  const canvas = canvasRef.value
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const sw = props.sourceWidth
  const sh = props.sourceHeight

  ctx.clearRect(0, 0, sw, sh)

  if (!cachedMaskCanvas || !cachedEdges || cachedEdgeCount === 0) return

  ctx.imageSmoothingEnabled = false
  ctx.drawImage(cachedMaskCanvas, 0, 0, sw, sh)

  const scale = props.effectiveScale
  if (scale <= 0) return

  const lineWidth = Math.min(BORDER_SCREEN_PX / scale, MAX_BORDER_CANVAS_PX)
  const edges = cachedEdges

  ctx.beginPath()
  for (let i = 0; i < cachedEdgeCount; i++) {
    const base = i * 4
    ctx.moveTo(edges[base]!, edges[base + 1]!)
    ctx.lineTo(edges[base + 2]!, edges[base + 3]!)
  }
  ctx.lineWidth = lineWidth
  ctx.strokeStyle = BORDER_COLOR
  ctx.lineCap = 'square'
  ctx.stroke()
}

watch(
  () => [props.regionMap, props.selectedRegionIds] as const,
  () => {
    buildMaskAndEdges()
    redraw()
  },
  { flush: 'post', immediate: true },
)

watch(
  () => props.effectiveScale,
  () => redraw(),
  { flush: 'post' },
)
</script>

<template>
  <canvas
    v-if="regionMap"
    ref="canvasRef"
    class="zoomable-image-viewport__layer"
    :width="sourceWidth"
    :height="sourceHeight"
    :style="{ transform, width: sourceWidth + 'px', height: sourceHeight + 'px' }"
  />
</template>
