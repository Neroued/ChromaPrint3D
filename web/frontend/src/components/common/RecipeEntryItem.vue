<script setup lang="ts">
import type { PaletteChannel } from '../../types'
import RecipeLayerBar from '../recipeEditor/RecipeLayerBar.vue'

defineProps<{
  recipe: number[]
  palette: PaletteChannel[]
  hex?: string
  lab?: [number, number, number]
}>()
</script>

<template>
  <div class="recipe-entry-item">
    <div class="recipe-entry-item__swatch" :style="{ backgroundColor: hex || '#888' }" />
    <div class="recipe-entry-item__info">
      <RecipeLayerBar :recipe="recipe" :palette="palette" />
      <span v-if="lab" class="recipe-entry-item__lab">
        L{{ lab[0].toFixed(1) }} a{{ lab[1].toFixed(1) }} b{{ lab[2].toFixed(1) }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.recipe-entry-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 3px 6px;
  border-radius: 4px;
  transition: background-color 0.15s;
}

.recipe-entry-item:hover {
  background-color: var(--n-color-hover, rgba(128, 128, 128, 0.08));
}

.recipe-entry-item__swatch {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  flex-shrink: 0;
  border: 1px solid rgba(128, 128, 128, 0.3);
}

.recipe-entry-item__info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.recipe-entry-item__lab {
  font-size: 11px;
  font-family: monospace;
  color: var(--n-text-color-3, #999);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
