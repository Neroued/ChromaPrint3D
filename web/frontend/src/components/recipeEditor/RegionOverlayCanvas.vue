<script setup lang="ts">
import { ref, watch } from 'vue'
import type { RegionMapData } from '../../composables/useRegionMap'

const props = defineProps<{
  regionMap: RegionMapData | null
  selectedRegionIds: Set<number>
  transform: string
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const MASK_ALPHA = 102

function redraw() {
  const canvas = canvasRef.value
  const map = props.regionMap
  if (!canvas || !map) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  if (props.selectedRegionIds.size === 0) {
    ctx.clearRect(0, 0, map.width, map.height)
    return
  }

  const imageData = ctx.createImageData(map.width, map.height)
  const data = imageData.data
  const regionIds = map.regionIds
  const selected = props.selectedRegionIds

  for (let i = 0; i < regionIds.length; i++) {
    const rid = regionIds[i]
    if (rid !== undefined && rid !== 0xffffffff && selected.has(rid)) continue
    data[i * 4 + 3] = MASK_ALPHA
  }

  ctx.putImageData(imageData, 0, 0)
}

watch(
  () => [props.regionMap, props.selectedRegionIds] as const,
  () => redraw(),
  { flush: 'post', immediate: true },
)
</script>

<template>
  <canvas
    v-if="regionMap"
    ref="canvasRef"
    class="zoomable-image-viewport__layer"
    :width="regionMap.width"
    :height="regionMap.height"
    :style="{ transform }"
  />
</template>
