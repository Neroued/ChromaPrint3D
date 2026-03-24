<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { NCard, NText, NVirtualList } from 'naive-ui'
import type { VirtualListInst } from 'naive-ui'
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

  const agg = new Map<number, { regionCount: number; pixelCount: number }>()
  for (const reg of s.regions) {
    const entry = agg.get(reg.recipe_index)
    if (entry) {
      entry.regionCount++
      entry.pixelCount += reg.pixel_count
    } else {
      agg.set(reg.recipe_index, { regionCount: 1, pixelCount: reg.pixel_count })
    }
  }

  const rows = s.unique_recipes.map((recipe, idx) => {
    const a = agg.get(idx) ?? { regionCount: 0, pixelCount: 0 }
    const pct = total > 0 ? ((a.pixelCount / total) * 100).toFixed(1) : '0.0'
    return { index: idx, recipe, ...a, areaPercent: pct }
  })
  rows.sort((a, b) => b.pixelCount - a.pixelCount)
  return rows
})

interface PairedRow {
  key: number
  left: RecipeRow
  right: RecipeRow | null
}

const pairedRows = computed<PairedRow[]>(() => {
  const pairs: PairedRow[] = []
  const rows = recipeRows.value
  for (let i = 0; i < rows.length; i += 2) {
    const left = rows[i]
    if (!left) continue
    pairs.push({ key: i, left, right: rows[i + 1] ?? null })
  }
  return pairs
})

const statsText = computed(() => {
  const rc = props.summary.unique_recipes.length
  const rg = props.summary.region_count
  return t('recipeEditor.summary.stats', { recipes: rc, regions: rg })
})

const virtualListRef = ref<VirtualListInst | null>(null)

function handleClick(index: number) {
  emit('selectRecipe', index)
}

watch(
  () => props.selectedRecipeIndex,
  (idx) => {
    if (idx === null) return
    const sortedPos = recipeRows.value.findIndex((r) => r.index === idx)
    if (sortedPos >= 0) {
      virtualListRef.value?.scrollTo({ index: Math.floor(sortedPos / 2) })
    }
  },
)
</script>

<template>
  <NCard size="small" :title="t('recipeEditor.summary.title')" class="recipe-summary-card">
    <template #header-extra>
      <NText depth="3" style="font-size: 12px">{{ statsText }}</NText>
    </template>
    <div class="recipe-summary-scroll-wrapper">
      <NVirtualList
        ref="virtualListRef"
        :items="pairedRows"
        :item-size="38"
        item-resizable
        class="recipe-summary-vlist"
      >
        <template #default="{ item }: { item: PairedRow }">
          <div class="recipe-summary-row">
            <div
              class="recipe-summary-item"
              :class="{ 'recipe-summary-item--selected': selectedRecipeIndex === item.left.index }"
              @click="handleClick(item.left.index)"
            >
              <div
                class="recipe-summary-item__color"
                :style="{ backgroundColor: item.left.recipe.hex }"
              />
              <div class="recipe-summary-item__info">
                <RecipeLayerBar :recipe="item.left.recipe.recipe" :palette="summary.palette" />
                <span class="recipe-summary-item__meta">
                  {{ t('recipeEditor.summary.regions', { count: item.left.regionCount }) }}
                  · {{ item.left.areaPercent }}%
                </span>
              </div>
              <NText
                v-if="item.left.recipe.from_model"
                depth="3"
                class="recipe-summary-item__badge"
                type="warning"
              >
                M
              </NText>
            </div>
            <div
              v-if="item.right"
              class="recipe-summary-item"
              :class="{
                'recipe-summary-item--selected': selectedRecipeIndex === item.right.index,
              }"
              @click="handleClick(item.right.index)"
            >
              <div
                class="recipe-summary-item__color"
                :style="{ backgroundColor: item.right.recipe.hex }"
              />
              <div class="recipe-summary-item__info">
                <RecipeLayerBar :recipe="item.right.recipe.recipe" :palette="summary.palette" />
                <span class="recipe-summary-item__meta">
                  {{ t('recipeEditor.summary.regions', { count: item.right.regionCount }) }}
                  · {{ item.right.areaPercent }}%
                </span>
              </div>
              <NText
                v-if="item.right.recipe.from_model"
                depth="3"
                class="recipe-summary-item__badge"
                type="warning"
              >
                M
              </NText>
            </div>
          </div>
        </template>
      </NVirtualList>
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

.recipe-summary-vlist {
  height: 100%;
}

.recipe-summary-row {
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
