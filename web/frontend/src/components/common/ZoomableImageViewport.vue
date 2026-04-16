<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted } from 'vue'
import { usePanZoom } from '../../composables/usePanZoom'
import type { PanZoomController } from '../../composables/usePanZoom'

const props = withDefaults(
  defineProps<{
    src?: string | null
    alt?: string
    height?: number | string
    checkerboard?: boolean
    controller?: PanZoomController
    showImage?: boolean
    contentWidth?: number
    contentHeight?: number
  }>(),
  {
    src: '',
    alt: 'preview',
    height: 300,
    checkerboard: true,
    controller: undefined,
    showImage: true,
    contentWidth: 0,
    contentHeight: 0,
  },
)

const containerRef = ref<HTMLElement | null>(null)
const containerW = ref(0)
const containerH = ref(0)

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  const el = containerRef.value
  if (!el) return
  containerW.value = el.clientWidth
  containerH.value = el.clientHeight
  resizeObserver = new ResizeObserver(() => {
    containerW.value = el.clientWidth
    containerH.value = el.clientHeight
  })
  resizeObserver.observe(el)
})

onUnmounted(() => {
  resizeObserver?.disconnect()
})

const useExplicitFit = computed(
  () =>
    props.contentWidth > 0 &&
    props.contentHeight > 0 &&
    containerW.value > 0 &&
    containerH.value > 0,
)

const fitParams = computed(() => {
  if (!useExplicitFit.value) return null
  const cw = containerW.value
  const ch = containerH.value
  const nw = props.contentWidth
  const nh = props.contentHeight
  const fitScale = Math.min(cw / nw, ch / nh)
  const offsetX = (cw - nw * fitScale) / 2
  const offsetY = (ch - nh * fitScale) / 2
  return { fitScale, offsetX, offsetY }
})

const internalPanZoom = usePanZoom()
const panZoom = computed<PanZoomController>(() => props.controller ?? internalPanZoom)

const effectiveScale = computed(() => {
  const fp = fitParams.value
  return fp ? fp.fitScale * panZoom.value.scale.value : panZoom.value.scale.value
})

const transformValue = computed(() => {
  const pz = panZoom.value
  const s = pz.scale.value
  const tx = pz.translateX.value
  const ty = pz.translateY.value
  const fp = fitParams.value
  if (fp) {
    const cx = tx + fp.offsetX * s
    const cy = ty + fp.offsetY * s
    const cs = fp.fitScale * s
    return `translate(${cx}px, ${cy}px) scale(${cs})`
  }
  return pz.previewTransform.value
})

const imgStyle = computed(() => {
  const base: Record<string, string> = { transform: transformValue.value }
  if (useExplicitFit.value) {
    base.width = `${props.contentWidth}px`
    base.height = `${props.contentHeight}px`
    base.objectFit = 'none'
  }
  return base
})

const imageSrc = computed(() => props.src ?? '')
const viewportStyle = computed(() => {
  const value = props.height
  return {
    height: typeof value === 'number' ? `${value}px` : value,
  }
})

const viewportClass = computed(() => ({
  'transparent-checkerboard-bg': props.checkerboard,
}))

function handleWheel(event: WheelEvent): void {
  panZoom.value.handleWheel(event)
}

function handleMouseDown(event: MouseEvent): void {
  panZoom.value.handleMouseDown(event)
}

function handleMouseMove(event: MouseEvent): void {
  panZoom.value.handleMouseMove(event)
}

function handleMouseUp(): void {
  panZoom.value.handleMouseUp()
}

function resetView(): void {
  panZoom.value.resetView()
}

defineExpose({
  resetView,
})
</script>

<template>
  <div
    ref="containerRef"
    class="zoomable-image-viewport"
    :class="viewportClass"
    :style="viewportStyle"
    @wheel="handleWheel"
    @mousedown="handleMouseDown"
    @mousemove="handleMouseMove"
    @mouseup="handleMouseUp"
    @mouseleave="handleMouseUp"
  >
    <img
      v-if="showImage && imageSrc"
      :src="imageSrc"
      class="zoomable-image-viewport__img"
      :style="imgStyle"
      draggable="false"
      :alt="alt"
    />
    <slot :transform="transformValue" :effective-scale="effectiveScale" />
  </div>
</template>

<style scoped>
.zoomable-image-viewport {
  position: relative;
  width: 100%;
  overflow: hidden;
  border: 1px solid var(--image-viewport-border-color, var(--n-border-color));
  border-radius: 4px;
  cursor: grab;
  user-select: none;
}

.zoomable-image-viewport:active {
  cursor: grabbing;
}

.zoomable-image-viewport__img {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  transform-origin: 0 0;
  pointer-events: none;
}

:slotted(.zoomable-image-viewport__layer) {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  transform-origin: 0 0;
  pointer-events: none;
}
</style>
