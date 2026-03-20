<script setup lang="ts">
import { computed } from 'vue'
import { NTooltip } from 'naive-ui'
import type { PaletteChannel } from '../../types'

const props = defineProps<{
  recipe: number[]
  palette: PaletteChannel[]
}>()

const layers = computed(() =>
  props.recipe.map((chIdx) => {
    const ch = props.palette[chIdx]
    return {
      hex: ch?.hex_color ?? '#888',
      initial: ch?.color?.charAt(0)?.toUpperCase() ?? '?',
      name: ch?.color ?? `ch${chIdx}`,
    }
  }),
)

const abbreviation = computed(() => layers.value.map((l) => l.initial).join('-'))

const fullName = computed(() => layers.value.map((l) => l.name).join(' - '))
</script>

<template>
  <NTooltip :delay="300">
    <template #trigger>
      <span class="recipe-layer-bar">
        <span class="recipe-layer-bar__colors">
          <span
            v-for="(layer, i) in layers"
            :key="i"
            class="recipe-layer-bar__segment"
            :style="{ backgroundColor: layer.hex }"
          />
        </span>
        <span class="recipe-layer-bar__abbr">{{ abbreviation }}</span>
      </span>
    </template>
    {{ fullName }}
  </NTooltip>
</template>

<style scoped>
.recipe-layer-bar {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
}

.recipe-layer-bar__colors {
  display: inline-flex;
  flex-shrink: 0;
  border-radius: 3px;
  overflow: hidden;
  border: 1px solid rgba(128, 128, 128, 0.3);
  height: 14px;
}

.recipe-layer-bar__segment {
  width: 8px;
  height: 100%;
}

.recipe-layer-bar__abbr {
  font-size: 12px;
  font-family: monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
