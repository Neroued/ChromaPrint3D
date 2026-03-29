import { computed, ref } from 'vue'
import type { ComputedRef, Ref } from 'vue'

type PanZoomOptions = {
  minScale?: number
  maxScale?: number
}

export type PanZoomController = {
  scale: Ref<number>
  translateX: Ref<number>
  translateY: Ref<number>
  dragging: Ref<boolean>
  previewTransform: ComputedRef<string>
  handleWheel: (e: WheelEvent) => void
  handleMouseDown: (e: MouseEvent) => void
  handleMouseMove: (e: MouseEvent) => void
  handleMouseUp: () => void
  resetView: () => void
}

export function usePanZoom(options: PanZoomOptions = {}) {
  const minScale = options.minScale ?? 0.1
  const maxScale = options.maxScale ?? 20

  const scale = ref(1)
  const translateX = ref(0)
  const translateY = ref(0)
  const dragging = ref(false)
  const lastMouse = ref({ x: 0, y: 0 })

  const previewTransform = computed(
    () => `translate(${translateX.value}px, ${translateY.value}px) scale(${scale.value})`,
  )

  function handleWheel(e: WheelEvent) {
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.1 : 0.9
    const newScale = Math.max(minScale, Math.min(maxScale, scale.value * factor))
    const ratio = newScale / scale.value

    const el = e.currentTarget as HTMLElement
    const rect = el.getBoundingClientRect()
    const mx = e.clientX - rect.left - el.clientLeft
    const my = e.clientY - rect.top - el.clientTop

    translateX.value = mx - (mx - translateX.value) * ratio
    translateY.value = my - (my - translateY.value) * ratio
    scale.value = newScale
  }

  function handleMouseDown(e: MouseEvent) {
    if (e.button !== 0) return
    dragging.value = true
    lastMouse.value = { x: e.clientX, y: e.clientY }
  }

  function handleMouseMove(e: MouseEvent) {
    if (!dragging.value) return
    translateX.value += e.clientX - lastMouse.value.x
    translateY.value += e.clientY - lastMouse.value.y
    lastMouse.value = { x: e.clientX, y: e.clientY }
  }

  function handleMouseUp() {
    dragging.value = false
  }

  function resetView() {
    scale.value = 1
    translateX.value = 0
    translateY.value = 0
  }

  return {
    scale,
    translateX,
    translateY,
    dragging,
    previewTransform,
    handleWheel,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    resetView,
  }
}
