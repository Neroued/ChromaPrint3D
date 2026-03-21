<script setup lang="ts">
import { computed } from 'vue'
import { NCard, NText, NStatistic, NGrid, NGi, NScrollbar } from 'naive-ui'
import { useI18n } from 'vue-i18n'
import type { ColorDBBuildResult, PaletteChannel, ColorDBEntry } from '../../types'
import RecipeEntryItem from '../common/RecipeEntryItem.vue'

const props = defineProps<{
  result: ColorDBBuildResult
  palette: PaletteChannel[]
}>()

const { t } = useI18n()

interface HueGroup {
  key: string
  label: string
  entries: ColorDBEntry[]
}

const HUE_GROUPS: { key: string; labelKey: string; min: number; max: number }[] = [
  { key: 'red', labelKey: 'colordb.result.hueRed', min: -15, max: 30 },
  { key: 'orange', labelKey: 'colordb.result.hueOrange', min: 30, max: 70 },
  { key: 'yellow', labelKey: 'colordb.result.hueYellow', min: 70, max: 105 },
  { key: 'green', labelKey: 'colordb.result.hueGreen', min: 105, max: 175 },
  { key: 'cyan', labelKey: 'colordb.result.hueCyan', min: 175, max: 210 },
  { key: 'blue', labelKey: 'colordb.result.hueBlue', min: 210, max: 280 },
  { key: 'purple', labelKey: 'colordb.result.huePurple', min: 280, max: 345 },
]

function labToHue(a: number, b: number): number {
  let h = (Math.atan2(b, a) * 180) / Math.PI
  if (h < 0) h += 360
  return h
}

function labToChroma(a: number, b: number): number {
  return Math.sqrt(a * a + b * b)
}

const groupedEntries = computed<HueGroup[]>(() => {
  const groups = new Map<string, ColorDBEntry[]>()
  groups.set('neutral', [])
  for (const g of HUE_GROUPS) groups.set(g.key, [])

  for (const entry of props.result.entries) {
    const [, a, b] = entry.lab
    const chroma = labToChroma(a, b)

    if (chroma < 8) {
      groups.get('neutral')!.push(entry)
      continue
    }

    const hue = labToHue(a, b)
    let assigned = false
    for (const g of HUE_GROUPS) {
      const { min, max } = g
      if (min < 0) {
        if (hue >= min + 360 || hue < max) {
          groups.get(g.key)!.push(entry)
          assigned = true
          break
        }
      } else if (hue >= min && hue < max) {
        groups.get(g.key)!.push(entry)
        assigned = true
        break
      }
    }
    if (!assigned) {
      groups.get('neutral')!.push(entry)
    }
  }

  for (const [, entries] of groups) {
    entries.sort((a, b) => b.lab[0] - a.lab[0])
  }

  const result: HueGroup[] = []
  for (const g of HUE_GROUPS) {
    const entries = groups.get(g.key) || []
    if (entries.length > 0) {
      result.push({ key: g.key, label: t(g.labelKey), entries })
    }
  }
  const neutral = groups.get('neutral') || []
  if (neutral.length > 0) {
    result.push({
      key: 'neutral',
      label: t('colordb.result.hueNeutral'),
      entries: neutral,
    })
  }
  return result
})
</script>

<template>
  <NCard :title="t('colordb.result.title')" size="small" class="colordb-result-card">
    <NGrid :cols="4" :x-gap="12" :y-gap="8" style="margin-bottom: 12px">
      <NGi>
        <NStatistic :label="t('colordb.result.name')" :value="result.name" />
      </NGi>
      <NGi>
        <NStatistic :label="t('colordb.result.channels')" :value="result.num_channels" />
      </NGi>
      <NGi>
        <NStatistic :label="t('colordb.result.entries')" :value="result.num_entries" />
      </NGi>
      <NGi>
        <NStatistic :label="t('colordb.result.colorLayers')" :value="result.max_color_layers" />
      </NGi>
    </NGrid>

    <NText depth="3" style="font-size: 13px; margin-bottom: 8px; display: block">
      {{ t('colordb.result.paletteLabel') }}:
      {{ palette.map((ch) => ch.color).join(' / ') }}
    </NText>

    <NScrollbar x-scrollable style="max-height: 450px">
      <div class="colordb-groups-row">
        <div v-for="group in groupedEntries" :key="group.key" class="colordb-group-col">
          <div class="colordb-group-header">
            {{ group.label }}
            <span class="colordb-group-count">({{ group.entries.length }})</span>
          </div>
          <RecipeEntryItem
            v-for="(entry, i) in group.entries"
            :key="i"
            :recipe="entry.recipe"
            :palette="palette"
            :hex="entry.hex"
            :lab="entry.lab"
          />
        </div>
      </div>
    </NScrollbar>
  </NCard>
</template>

<style scoped>
.colordb-result-card {
  max-width: 100%;
}

.colordb-groups-row {
  display: flex;
  gap: 8px;
  align-items: flex-start;
}

.colordb-group-col {
  flex: 1 0 150px;
  max-width: 220px;
}

.colordb-group-header {
  font-weight: 600;
  font-size: 13px;
  padding: 2px 6px 6px;
  border-bottom: 1px solid rgba(128, 128, 128, 0.2);
  margin-bottom: 2px;
  white-space: nowrap;
}

.colordb-group-count {
  font-weight: 400;
  color: var(--n-text-color-3, #999);
}
</style>
