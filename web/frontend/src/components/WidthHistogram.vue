<script setup lang="ts">
import { NStatistic, NTag, NText } from 'naive-ui'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  Chart,
  BarElement,
  BarController,
  CategoryScale,
  LinearScale,
  Tooltip,
  type ChartConfiguration,
} from 'chart.js'
import type { ShapeWidthInfo } from '../types'

Chart.register(BarElement, BarController, CategoryScale, LinearScale, Tooltip)

const props = defineProps<{
  shapes: ShapeWidthInfo[]
  nozzleDiameter: number
  imageWidthMm: number
  imageHeightMm: number
  targetWidthMm: number
  targetHeightMm: number
}>()

const { t } = useI18n()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chartInstance: Chart | null = null

const BIN_WIDTH = 0.05

const displayScale = computed(() => {
  const tw = props.targetWidthMm
  const th = props.targetHeightMm
  if (tw <= 0 && th <= 0) return 1
  if (props.imageWidthMm <= 0 || props.imageHeightMm <= 0) return 1
  const sw = tw > 0 ? tw / props.imageWidthMm : Infinity
  const sh = th > 0 ? th / props.imageHeightMm : Infinity
  return Math.min(sw, sh)
})

const histogramData = computed(() => {
  if (props.shapes.length === 0)
    return { labels: [] as string[], counts: [] as number[], colors: [] as string[] }

  const scale = displayScale.value
  const widths = props.shapes.map((s) => s.min_width_mm * scale)
  const maxW = Math.max(...widths)
  const numBins = Math.max(1, Math.ceil(maxW / BIN_WIDTH))

  const counts = new Array<number>(numBins).fill(0)
  for (const w of widths) {
    const bin = Math.min(Math.floor(w / BIN_WIDTH), numBins - 1)
    counts[bin] = (counts[bin] ?? 0) + 1
  }

  const labels = counts.map((_, i) => (i * BIN_WIDTH).toFixed(2))

  const nozzleBin = Math.floor(props.nozzleDiameter / BIN_WIDTH)
  const colors = counts.map((_, i) =>
    i < nozzleBin ? 'rgba(232, 85, 85, 0.75)' : 'rgba(66, 135, 245, 0.75)',
  )

  return { labels, counts, colors }
})

const belowNozzleCount = computed(
  () =>
    props.shapes.filter((s) => s.min_width_mm * displayScale.value < props.nozzleDiameter).length,
)

const scaleSq = computed(() => displayScale.value * displayScale.value)

const totalArea = computed(() =>
  props.shapes.reduce((sum, s) => sum + s.area_mm2 * scaleSq.value, 0),
)

const belowNozzleArea = computed(() =>
  props.shapes
    .filter((s) => s.min_width_mm * displayScale.value < props.nozzleDiameter)
    .reduce((sum, s) => sum + s.area_mm2 * scaleSq.value, 0),
)

const belowNozzleAreaPercent = computed(() => {
  if (totalArea.value <= 0) return 0
  return Math.round((belowNozzleArea.value / totalArea.value) * 100)
})

const globalMinWidth = computed(() => {
  if (props.shapes.length === 0) return 0
  return Math.min(...props.shapes.map((s) => s.min_width_mm)) * displayScale.value
})

const suggestedScale = computed(() => {
  if (globalMinWidth.value <= 0 || globalMinWidth.value >= props.nozzleDiameter) return 1
  const factor = props.nozzleDiameter / globalMinWidth.value
  return Math.ceil(factor * 10) / 10
})

const nozzleLinePlugin = {
  id: 'nozzleLine',
  afterDraw(chart: Chart) {
    const xScale = chart.scales['x']
    const yScale = chart.scales['y']
    if (!xScale || !yScale) return

    const nozzleBinIndex = props.nozzleDiameter / BIN_WIDTH
    const xPos = xScale.getPixelForValue(nozzleBinIndex)
    const ctx = chart.ctx

    ctx.save()
    ctx.setLineDash([6, 3])
    ctx.strokeStyle = 'rgba(232, 85, 85, 1)'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(xPos, yScale.top)
    ctx.lineTo(xPos, yScale.bottom)
    ctx.stroke()

    ctx.setLineDash([])
    ctx.fillStyle = 'rgba(232, 85, 85, 1)'
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(`${props.nozzleDiameter}mm`, xPos, yScale.top - 6)
    ctx.restore()
  },
}

function buildChart() {
  if (!canvasRef.value) return
  if (chartInstance) {
    chartInstance.destroy()
    chartInstance = null
  }

  const { labels, counts, colors } = histogramData.value
  if (labels.length === 0) return

  const config: ChartConfiguration<'bar'> = {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          data: counts,
          backgroundColor: colors,
          borderWidth: 0,
          barPercentage: 1.0,
          categoryPercentage: 1.0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: {
          callbacks: {
            title(items) {
              const i = items[0]?.dataIndex ?? 0
              const lo = (i * BIN_WIDTH).toFixed(2)
              const hi = ((i + 1) * BIN_WIDTH).toFixed(2)
              return `${lo} – ${hi} mm`
            },
            label(item) {
              return `${item.raw} ${t('widthAnalysis.shapesUnit')}`
            },
          },
        },
        legend: { display: false },
      },
      scales: {
        x: {
          title: { display: true, text: t('widthAnalysis.xAxis') },
          ticks: {
            maxTicksLimit: 15,
            callback(val) {
              const idx = typeof val === 'number' ? val : parseInt(val as string, 10)
              if (idx % 4 === 0) return (idx * BIN_WIDTH).toFixed(1)
              return ''
            },
          },
        },
        y: {
          title: { display: true, text: t('widthAnalysis.yAxis') },
          beginAtZero: true,
          ticks: { precision: 0 },
        },
      },
    },
    plugins: [nozzleLinePlugin],
  }

  chartInstance = new Chart(canvasRef.value, config)
}

watch(histogramData, buildChart)

onMounted(buildChart)
onUnmounted(() => {
  if (chartInstance) {
    chartInstance.destroy()
    chartInstance = null
  }
})
</script>

<template>
  <div class="width-histogram">
    <div class="width-histogram__chart-wrap">
      <canvas ref="canvasRef" />
    </div>

    <div class="width-histogram__stats">
      <NStatistic :label="t('widthAnalysis.minWidth')">
        <NText :type="globalMinWidth < nozzleDiameter ? 'error' : 'success'">
          {{ globalMinWidth.toFixed(3) }} mm
        </NText>
      </NStatistic>

      <NStatistic :label="t('widthAnalysis.belowNozzle')">
        <NTag :type="belowNozzleCount > 0 ? 'error' : 'success'" size="small">
          {{ belowNozzleCount }} / {{ shapes.length }}
        </NTag>
      </NStatistic>

      <NStatistic :label="t('widthAnalysis.belowNozzleArea')">
        <NTag :type="belowNozzleAreaPercent > 0 ? 'error' : 'success'" size="small">
          {{ belowNozzleArea.toFixed(1) }} mm² ({{ belowNozzleAreaPercent }}%)
        </NTag>
      </NStatistic>

      <NStatistic v-if="suggestedScale > 1" :label="t('widthAnalysis.suggestedScale')">
        <NText type="warning">{{ suggestedScale.toFixed(1) }}x</NText>
      </NStatistic>
    </div>
  </div>
</template>

<style scoped>
.width-histogram {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.width-histogram__chart-wrap {
  position: relative;
  height: 180px;
}

.width-histogram__stats {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
}
</style>
