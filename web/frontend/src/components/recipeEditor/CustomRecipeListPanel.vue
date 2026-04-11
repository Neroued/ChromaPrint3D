<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { NButton, NText, NScrollbar } from 'naive-ui'
import type { RecipeCandidate, PaletteChannel } from '../../types'
import RecipeLayerBar from './RecipeLayerBar.vue'

defineProps<{
  items: RecipeCandidate[]
  palette: PaletteChannel[]
  hasSelection: boolean
}>()

const emit = defineEmits<{
  select: [candidate: RecipeCandidate]
  create: []
}>()

const { t } = useI18n()
</script>

<template>
  <div class="custom-recipe-list-panel">
    <NButton
      size="small"
      dashed
      style="width: 100%; flex-shrink: 0"
      :disabled="!hasSelection"
      @click="emit('create')"
    >
      {{ t('recipeEditor.customRecipe.createNew') }}
    </NButton>

    <div class="custom-recipe-list-scroll">
      <NScrollbar>
        <div v-if="items.length === 0" class="custom-recipe-list-empty">
          <NText depth="3">{{ t('recipeEditor.customRecipe.emptyHistory') }}</NText>
        </div>
        <div v-else class="custom-recipe-list">
          <div
            v-for="(item, idx) in items"
            :key="item.recipe.join('-') + '-' + idx"
            class="custom-recipe-item"
            @click="emit('select', item)"
          >
            <div class="custom-recipe-item__color" :style="{ backgroundColor: item.hex }" />
            <div class="custom-recipe-item__info">
              <RecipeLayerBar :recipe="item.recipe" :palette="palette" />
              <span class="custom-recipe-item__hex">{{ item.hex }}</span>
            </div>
          </div>
        </div>
      </NScrollbar>
    </div>
  </div>
</template>

<style scoped>
.custom-recipe-list-panel {
  display: flex;
  flex-direction: column;
  gap: 8px;
  height: 100%;
  min-height: 0;
}

.custom-recipe-list-scroll {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.custom-recipe-list-empty {
  padding: 32px 16px;
  text-align: center;
}

.custom-recipe-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.custom-recipe-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.custom-recipe-item:hover {
  background-color: var(--n-color-hover);
}

.custom-recipe-item__color {
  width: 28px;
  height: 28px;
  border-radius: 4px;
  flex-shrink: 0;
  border: 1px solid rgba(128, 128, 128, 0.3);
}

.custom-recipe-item__info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.custom-recipe-item__hex {
  font-size: 11px;
  font-family: monospace;
  color: var(--n-text-color-3, #999);
}
</style>
