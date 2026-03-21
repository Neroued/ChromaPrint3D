<script setup lang="ts">
import { computed, ref, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import { NCard, NButton, NSpace, NText, NScrollbar, NSelect, NSpin, NColorPicker } from 'naive-ui'
import type { RecipeCandidate, LabColor, PaletteChannel } from '../../types'
import { fetchRecipeAlternatives } from '../../api/recipeEditor'
import { hexToLab } from '../../utils/colorConvert'
import RecipeLayerBar from './RecipeLayerBar.vue'

const props = defineProps<{
  taskId: string
  targetLab: LabColor | null
  targetHex: string | null
  palette: PaletteChannel[]
}>()

const emit = defineEmits<{
  select: [candidate: RecipeCandidate]
}>()

const { t } = useI18n()
const candidates = ref<RecipeCandidate[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const sortBy = ref<'delta_e76' | 'lightness_diff' | 'hue_diff'>('delta_e76')
const hasMore = ref(false)
const scrollWrapperRef = ref<HTMLElement | null>(null)

const customColorHex = ref<string | null>(null)
const customTargetLab = ref<LabColor | null>(null)
const pickerValue = ref('#FFFFFF')

const hasEyeDropper = typeof window !== 'undefined' && 'EyeDropper' in window

const effectiveTargetLab = computed<LabColor | null>(() => customTargetLab.value ?? props.targetLab)

const PAGE_SIZE = 10

const sortOptions = computed(() => [
  { label: t('recipeEditor.candidates.sortDeltaE'), value: 'delta_e76' },
  { label: t('recipeEditor.candidates.sortLightness'), value: 'lightness_diff' },
  { label: t('recipeEditor.candidates.sortHue'), value: 'hue_diff' },
])

const sortedCandidates = computed(() => {
  const list = [...candidates.value]
  const key = sortBy.value
  list.sort((a, b) => Math.abs(a[key]) - Math.abs(b[key]))
  return list
})

function applyCustomColor(hex: string) {
  customColorHex.value = hex
  customTargetLab.value = hexToLab(hex)
}

function handlePickerComplete(hex: string) {
  applyCustomColor(hex)
}

function clearCustomColor() {
  customColorHex.value = null
  customTargetLab.value = null
  pickerValue.value = props.targetHex ?? '#FFFFFF'
}

async function pickColorFromScreen() {
  try {
    const eyeDropper = new (
      window as unknown as { EyeDropper: new () => { open(): Promise<{ sRGBHex: string }> } }
    ).EyeDropper()
    const result = await eyeDropper.open()
    applyCustomColor(result.sRGBHex)
    pickerValue.value = result.sRGBHex
  } catch {
    /* user cancelled */
  }
}

async function loadCandidates(append = false) {
  const lab = effectiveTargetLab.value
  if (!lab || !props.taskId) return
  loading.value = true
  error.value = null
  try {
    const offset = append ? candidates.value.length : 0
    const result = await fetchRecipeAlternatives(props.taskId, lab, PAGE_SIZE, offset)
    if (append) {
      candidates.value = [...candidates.value, ...result]
    } else {
      candidates.value = result
    }
    hasMore.value = result.length >= PAGE_SIZE
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
  if (hasMore.value) {
    void nextTick(() => autoLoadIfNeeded())
  }
}

function autoLoadIfNeeded() {
  if (!scrollWrapperRef.value || !hasMore.value || loading.value) return
  const container = scrollWrapperRef.value.querySelector('.n-scrollbar-container')
  if (!container) return
  if (container.scrollHeight <= container.clientHeight + 10) {
    void loadCandidates(true)
  }
}

function handleScrollbarScroll(e: Event) {
  const el = e.target as HTMLElement | null
  if (!el || !hasMore.value || loading.value) return
  if (el.scrollHeight - el.scrollTop - el.clientHeight < 100) {
    void loadCandidates(true)
  }
}

watch(
  () => props.targetHex,
  (hex) => {
    if (!hex) {
      customColorHex.value = null
      customTargetLab.value = null
      pickerValue.value = '#FFFFFF'
    } else if (!customColorHex.value) {
      pickerValue.value = hex
    }
  },
  { immediate: true },
)

watch(
  [effectiveTargetLab, () => props.taskId],
  () => {
    candidates.value = []
    hasMore.value = false
    if (effectiveTargetLab.value) {
      void loadCandidates()
    }
  },
  { immediate: true, deep: true },
)

function handleSelect(candidate: RecipeCandidate) {
  emit('select', candidate)
}
</script>

<template>
  <NCard size="small" :title="t('recipeEditor.candidates.title')" class="candidate-card">
    <template #header-extra>
      <NSpace :size="6" align="center">
        <NText depth="3" style="font-size: 12px; white-space: nowrap">
          {{ t('recipeEditor.candidates.sortLabel') }}
        </NText>
        <NSelect
          v-model:value="sortBy"
          size="small"
          :options="sortOptions"
          style="width: 150px"
          :consistent-menu-width="false"
        />
      </NSpace>
    </template>
    <div v-if="targetLab" class="candidate-toolbar">
      <NSpace :size="8" align="center">
        <NColorPicker
          v-model:value="pickerValue"
          size="small"
          :modes="['hex']"
          :show-alpha="false"
          style="width: 100px"
          @complete="handlePickerComplete"
        />
        <NButton
          v-if="hasEyeDropper"
          size="tiny"
          quaternary
          :title="t('recipeEditor.candidates.eyeDropper')"
          @click="pickColorFromScreen"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="m2 22 1-1h3l9-9" />
            <path d="M3 21v-3l9-9" />
            <path
              d="m15 6 3.4-3.4a2.1 2.1 0 1 1 3 3L18 9l.4.4a2.1 2.1 0 1 1-3 3l-3.8-3.8a2.1 2.1 0 1 1 3-3L15 6"
            />
          </svg>
        </NButton>
        <NButton v-if="customColorHex" size="tiny" quaternary @click="clearCustomColor">
          {{ t('recipeEditor.candidates.clearCustomColor') }}
        </NButton>
      </NSpace>
    </div>
    <div ref="scrollWrapperRef" class="candidate-scroll-wrapper">
      <NSpin :show="loading && candidates.length === 0" style="height: 100%">
        <NScrollbar @scroll="handleScrollbarScroll">
          <div v-if="candidates.length === 0 && !loading" class="candidate-empty">
            <NText depth="3">{{ t('recipeEditor.candidates.empty') }}</NText>
          </div>
          <div v-else class="candidate-list">
            <div
              v-for="(c, idx) in sortedCandidates"
              :key="c.recipe.join('-') + '-' + idx"
              class="candidate-item"
              @click="handleSelect(c)"
            >
              <div class="candidate-item__color" :style="{ backgroundColor: c.hex }" />
              <div class="candidate-item__info">
                <RecipeLayerBar :recipe="c.recipe" :palette="palette" />
                <span class="candidate-item__metrics">
                  ΔE {{ c.delta_e76.toFixed(1) }} · ΔL {{ c.lightness_diff.toFixed(1) }} · ΔH
                  {{ c.hue_diff.toFixed(1) }}
                </span>
              </div>
              <NText v-if="c.from_model" depth="3" class="candidate-item__badge" type="warning">
                M
              </NText>
            </div>
          </div>
          <div v-if="loading && candidates.length > 0" class="candidate-loading-more">
            <NSpin :size="18" />
          </div>
        </NScrollbar>
      </NSpin>
    </div>
    <NText v-if="error" type="error" class="candidate-error">{{ error }}</NText>
  </NCard>
</template>

<style scoped>
.candidate-card {
  height: 100%;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.candidate-card :deep(.n-card__content) {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.candidate-scroll-wrapper {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.candidate-scroll-wrapper :deep(.n-spin-content) {
  height: 100%;
}

.candidate-toolbar {
  margin-bottom: 8px;
}

.candidate-list {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
}

.candidate-empty {
  padding: 24px;
  text-align: center;
}

.candidate-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 6px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.candidate-item:hover {
  background-color: var(--n-color-hover);
}

.candidate-item__color {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  flex-shrink: 0;
  border: 1px solid rgba(128, 128, 128, 0.3);
}

.candidate-item__info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.candidate-item__recipe {
  font-size: 13px;
  font-family: monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.candidate-item__metrics {
  font-size: 11px;
  color: var(--n-text-color-3, #999);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.candidate-item__badge {
  font-size: 11px;
  flex-shrink: 0;
}

.candidate-loading-more {
  padding: 8px;
  text-align: center;
}

.candidate-error {
  display: block;
  margin-top: 8px;
  font-size: 12px;
}
</style>
