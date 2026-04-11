<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useMessage } from 'naive-ui'
import {
  NModal,
  NCard,
  NButton,
  NSpace,
  NSelect,
  NInputNumber,
  NColorPicker,
  NAlert,
  NText,
  NTooltip,
} from 'naive-ui'
import type { LabColor, PaletteChannel } from '../../types'
import { hexToLab, labToHex } from '../../utils/colorConvert'
import { predictRecipeColor } from '../../api/recipeEditor'

const MAX_COLOR_LAYERS = 10

const props = defineProps<{
  show: boolean
  taskId: string
  initialRecipe: number[]
  initialHex: string
  palette: PaletteChannel[]
  currentColorLayers: number
}>()

const emit = defineEmits<{
  'update:show': [value: boolean]
  confirm: [payload: { recipe: number[]; mappedLab: LabColor; hex: string; fromModel: boolean }]
}>()

const { t } = useI18n()
const message = useMessage()

const layers = ref<number[]>([])
const hexColor = ref('#FFFFFF')
const labL = ref(100)
const labA = ref(0)
const labB = ref(0)
const showUpgradeWarning = ref(false)
const predicting = ref(false)
const colorFromModel = ref(false)
let suppressColorSync = false

const channelOptions = computed(() =>
  props.palette.map((ch, idx) => ({
    label: ch.color || `ch${idx}`,
    value: idx,
  })),
)

const defaultChannel = computed(() => {
  const idx = props.palette.findIndex(
    (ch) => ch.color?.toLowerCase() === 'white' || ch.color?.charAt(0)?.toLowerCase() === 'w',
  )
  return idx >= 0 ? idx : 0
})

const canAddLayer = computed(() => layers.value.length < MAX_COLOR_LAYERS)
const canRemoveLayer = computed(() => layers.value.length > 1)
const willUpgrade = computed(() => layers.value.length > props.currentColorLayers)
const isValid = computed(
  () =>
    layers.value.length >= 1 && layers.value.every((ch) => ch >= 0 && ch < props.palette.length),
)

watch(
  () => props.show,
  (visible) => {
    if (visible) {
      layers.value = [...props.initialRecipe]
      hexColor.value = props.initialHex || '#FFFFFF'
      syncLabFromHex(hexColor.value)
      showUpgradeWarning.value = false
    }
  },
)

function syncLabFromHex(hex: string) {
  suppressColorSync = true
  const lab = hexToLab(hex)
  labL.value = Math.round(lab.L * 100) / 100
  labA.value = Math.round(lab.a * 100) / 100
  labB.value = Math.round(lab.b * 100) / 100
  suppressColorSync = false
}

function syncHexFromLab() {
  if (suppressColorSync) return
  suppressColorSync = true
  hexColor.value = labToHex({ L: labL.value, a: labA.value, b: labB.value })
  suppressColorSync = false
  colorFromModel.value = false
}

function handleHexChange(hex: string) {
  hexColor.value = hex
  syncLabFromHex(hex)
  colorFromModel.value = false
}

async function handlePredict() {
  if (!isValid.value || predicting.value) return
  predicting.value = true
  try {
    const result = await predictRecipeColor(props.taskId, [...layers.value])
    hexColor.value = result.hex
    syncLabFromHex(result.hex)
    colorFromModel.value = true
  } catch (e: unknown) {
    colorFromModel.value = false
    const msg = e instanceof Error ? e.message : String(e)
    if (msg.includes('no_forward_model') || msg.includes('501')) {
      message.warning(t('recipeEditor.customRecipe.noForwardModel'))
    } else {
      message.error(t('recipeEditor.customRecipe.predictFailed'))
    }
  } finally {
    predicting.value = false
  }
}

function addLayer() {
  if (!canAddLayer.value) return
  layers.value.push(defaultChannel.value)
}

function removeLayer(index: number) {
  if (!canRemoveLayer.value) return
  layers.value.splice(index, 1)
}

function moveLayerUp(index: number) {
  if (index <= 0) return
  const arr = layers.value
  ;[arr[index - 1], arr[index]] = [arr[index]!, arr[index - 1]!]
}

function moveLayerDown(index: number) {
  if (index >= layers.value.length - 1) return
  const arr = layers.value
  ;[arr[index], arr[index + 1]] = [arr[index + 1]!, arr[index]!]
}

function handleConfirm() {
  if (!isValid.value) return
  if (willUpgrade.value && !showUpgradeWarning.value) {
    showUpgradeWarning.value = true
    return
  }
  emit('confirm', {
    recipe: [...layers.value],
    mappedLab: { L: labL.value, a: labA.value, b: labB.value },
    hex: hexColor.value,
    fromModel: colorFromModel.value,
  })
  emit('update:show', false)
}

function handleClose() {
  emit('update:show', false)
}

function getChannelHex(chIdx: number): string {
  return props.palette[chIdx]?.hex_color ?? '#888'
}

function renderChannelLabel(option: { label: string; value: number }) {
  return option.label
}
</script>

<template>
  <NModal :show="show" @update:show="$emit('update:show', $event)">
    <NCard
      :title="t('recipeEditor.customRecipe.title')"
      :bordered="false"
      size="medium"
      role="dialog"
      :aria-modal="true"
      style="width: 480px; max-width: 95vw"
      :closable="true"
      @close="handleClose"
    >
      <div class="custom-recipe-dialog">
        <div class="section-label">
          <NText strong>{{ t('recipeEditor.customRecipe.layersLabel') }}</NText>
          <NText depth="3" style="font-size: 12px; margin-left: 8px">
            {{ t('recipeEditor.customRecipe.layerCount', { count: layers.length }) }}
          </NText>
        </div>

        <div class="layer-list">
          <div v-for="(chIdx, i) in layers" :key="i" class="layer-row">
            <span class="layer-index">{{ i + 1 }}</span>
            <span class="layer-swatch" :style="{ backgroundColor: getChannelHex(chIdx) }" />
            <NSelect
              :value="chIdx"
              :options="channelOptions"
              :render-label="renderChannelLabel"
              size="small"
              style="flex: 1; min-width: 0"
              @update:value="(v: number) => (layers[i] = v)"
            />
            <NSpace :size="2">
              <NTooltip>
                <template #trigger>
                  <NButton size="tiny" quaternary :disabled="i === 0" @click="moveLayerUp(i)">
                    ↑
                  </NButton>
                </template>
                {{ t('recipeEditor.customRecipe.moveUp') }}
              </NTooltip>
              <NTooltip>
                <template #trigger>
                  <NButton
                    size="tiny"
                    quaternary
                    :disabled="i === layers.length - 1"
                    @click="moveLayerDown(i)"
                  >
                    ↓
                  </NButton>
                </template>
                {{ t('recipeEditor.customRecipe.moveDown') }}
              </NTooltip>
              <NTooltip>
                <template #trigger>
                  <NButton
                    size="tiny"
                    quaternary
                    :disabled="!canRemoveLayer"
                    @click="removeLayer(i)"
                  >
                    ✕
                  </NButton>
                </template>
                {{ t('recipeEditor.customRecipe.removeLayer') }}
              </NTooltip>
            </NSpace>
          </div>
        </div>

        <NButton
          size="small"
          dashed
          :disabled="!canAddLayer"
          style="width: 100%; margin-top: 4px"
          @click="addLayer"
        >
          {{ t('recipeEditor.customRecipe.addLayer') }}
          <NText v-if="!canAddLayer" depth="3" style="font-size: 11px; margin-left: 6px">
            ({{ t('recipeEditor.customRecipe.maxLayers', { max: MAX_COLOR_LAYERS }) }})
          </NText>
        </NButton>

        <div class="section-label" style="margin-top: 16px">
          <NText strong>{{ t('recipeEditor.customRecipe.previewColor') }}</NText>
          <NButton
            size="tiny"
            type="info"
            secondary
            :loading="predicting"
            :disabled="!isValid || predicting"
            style="margin-left: auto"
            @click="handlePredict"
          >
            {{
              predicting
                ? t('recipeEditor.customRecipe.predicting')
                : t('recipeEditor.customRecipe.predictColor')
            }}
          </NButton>
        </div>

        <div class="color-section">
          <div class="color-picker-row">
            <NColorPicker
              :value="hexColor"
              size="small"
              :modes="['hex']"
              :show-alpha="false"
              style="width: 120px"
              @complete="handleHexChange"
            />
            <span class="color-preview-swatch" :style="{ backgroundColor: hexColor }" />
          </div>
          <div class="lab-inputs">
            <div class="lab-field">
              <NText depth="3" style="font-size: 12px">L</NText>
              <NInputNumber
                v-model:value="labL"
                size="small"
                :min="0"
                :max="100"
                :step="1"
                style="width: 90px"
                @update:value="syncHexFromLab"
              />
            </div>
            <div class="lab-field">
              <NText depth="3" style="font-size: 12px">a</NText>
              <NInputNumber
                v-model:value="labA"
                size="small"
                :min="-128"
                :max="127"
                :step="1"
                style="width: 90px"
                @update:value="syncHexFromLab"
              />
            </div>
            <div class="lab-field">
              <NText depth="3" style="font-size: 12px">b</NText>
              <NInputNumber
                v-model:value="labB"
                size="small"
                :min="-128"
                :max="127"
                :step="1"
                style="width: 90px"
                @update:value="syncHexFromLab"
              />
            </div>
          </div>
        </div>

        <NAlert v-if="showUpgradeWarning && willUpgrade" type="warning" style="margin-top: 12px">
          {{
            t('recipeEditor.customRecipe.layerUpgradeWarning', {
              old: currentColorLayers,
              new: layers.length,
            })
          }}
        </NAlert>
      </div>

      <template #footer>
        <NSpace justify="end">
          <NButton @click="handleClose">
            {{ t('common.cancel') }}
          </NButton>
          <NButton type="primary" :disabled="!isValid" @click="handleConfirm">
            {{
              showUpgradeWarning && willUpgrade
                ? t('recipeEditor.customRecipe.confirmUpgrade')
                : t('recipeEditor.customRecipe.apply')
            }}
          </NButton>
        </NSpace>
      </template>
    </NCard>
  </NModal>
</template>

<style scoped>
.custom-recipe-dialog {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.section-label {
  display: flex;
  align-items: baseline;
  margin-bottom: 4px;
}

.layer-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 280px;
  overflow-y: auto;
}

.layer-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.layer-index {
  width: 18px;
  text-align: center;
  font-size: 12px;
  color: var(--n-text-color-3, #999);
  flex-shrink: 0;
}

.layer-swatch {
  width: 16px;
  height: 16px;
  border-radius: 3px;
  border: 1px solid rgba(128, 128, 128, 0.3);
  flex-shrink: 0;
}

.color-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.color-picker-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.color-preview-swatch {
  width: 32px;
  height: 32px;
  border-radius: 4px;
  border: 1px solid rgba(128, 128, 128, 0.3);
  flex-shrink: 0;
}

.lab-inputs {
  display: flex;
  gap: 12px;
}

.lab-field {
  display: flex;
  align-items: center;
  gap: 4px;
}
</style>
