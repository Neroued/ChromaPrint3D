<script setup lang="ts">
import { computed, ref, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import { NCard, NText, NScrollbar } from 'naive-ui'
import type { RecipeEditorSummary, RecipeInfo } from '../../types'
import RecipeLayerBar from './RecipeLayerBar.vue'

const props = defineProps<{
  summary: RecipeEditorSummary
  selectedRecipeIndex: number | null
}>()

const emit = defineEmits<{
  selectRecipe: [recipeIndex: number]
}>()

const { t } = useI18n()

const totalPixels = computed(() => props.summary.width * props.summary.height)

interface RecipeRow {
  index: number
  recipe: RecipeInfo
  regionCount: number
  pixelCount: number
  areaPercent: string
}

const recipeRows = computed<RecipeRow[]>(() => {
  const s = props.summary
  const total = totalPixels.value
  const rows = s.unique_recipes.map((recipe, idx) => {
    let regionCount = 0
    let pixelCount = 0
    for (const reg of s.regions) {
      if (reg.recipe_index === idx) {
        regionCount++
        pixelCount += reg.pixel_count
      }
    }
    const pct = total > 0 ? ((pixelCount / total) * 100).toFixed(1) : '0.0'
    return { index: idx, recipe, regionCount, pixelCount, areaPercent: pct }
  })
  rows.sort((a, b) => b.pixelCount - a.pixelCount)
  return rows
})

const statsText = computed(() => {
  const rc = props.summary.unique_recipes.length
  const rg = props.summary.region_count
  return t('recipeEditor.summary.stats', { recipes: rc, regions: rg })
})

const listRef = ref<HTMLElement | null>(null)

function handleClick(index: number) {
  emit('selectRecipe', index)
}

watch(
  () => props.selectedRecipeIndex,
  (idx) => {
    if (idx === null || !listRef.value) return
    void nextTick(() => {
      const el = listRef.value?.querySelector(`[data-recipe-index="${idx}"]`)
      if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    })
  },
)
</script>

<template>
  <NCard size="small" :title="t('recipeEditor.summary.title')" class="recipe-summary-card">
    <template #header-extra>
      <NText depth="3" style="font-size: 12px">{{ statsText }}</NText>
    </template>
    <div class="recipe-summary-scroll-wrapper">
      <NScrollbar>
        <div ref="listRef" class="recipe-summary-list">
          <div
            v-for="row in recipeRows"
            :key="row.index"
            :data-recipe-index="row.index"
            class="recipe-summary-item"
            :class="{ 'recipe-summary-item--selected': selectedRecipeIndex === row.index }"
            @click="handleClick(row.index)"
          >
            <div class="recipe-summary-item__color" :style="{ backgroundColor: row.recipe.hex }" />
            <div class="recipe-summary-item__info">
              <RecipeLayerBar :recipe="row.recipe.recipe" :palette="summary.palette" />
              <span class="recipe-summary-item__meta">
                {{ t('recipeEditor.summary.regions', { count: row.regionCount }) }}
                · {{ row.areaPercent }}%
              </span>
            </div>
            <NText
              v-if="row.recipe.from_model"
              depth="3"
              class="recipe-summary-item__badge"
              type="warning"
            >
              M
            </NText>
          </div>
        </div>
      </NScrollbar>
    </div>
  </NCard>
</template>

<style scoped>
.recipe-summary-card {
  height: 100%;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.recipe-summary-card :deep(.n-card__content) {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.recipe-summary-scroll-wrapper {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.recipe-summary-list {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2px;
}

.recipe-summary-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 6px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.recipe-summary-item:hover {
  background-color: var(--n-color-hover);
}

.recipe-summary-item--selected {
  background-color: var(--n-color-active, rgba(24, 160, 88, 0.1));
  box-shadow: inset 0 0 0 2px var(--n-color-target, #18a058);
}

.recipe-summary-item__color {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  flex-shrink: 0;
  border: 1px solid rgba(128, 128, 128, 0.3);
}

.recipe-summary-item__info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.recipe-summary-item__label {
  font-size: 13px;
  font-family: monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.recipe-summary-item__meta {
  font-size: 11px;
  color: var(--n-text-color-3, #999);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.recipe-summary-item__badge {
  font-size: 11px;
  flex-shrink: 0;
}
</style>
