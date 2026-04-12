<script setup lang="ts">
import {
  NAlert,
  NButton,
  NCard,
  NCollapse,
  NCollapseItem,
  NFormItem,
  NInputNumber,
  NRadioButton,
  NRadioGroup,
  NSelect,
  NSlider,
  NSpace,
  NSpin,
  NSwitch,
  NText,
  NTooltip,
} from 'naive-ui'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useParamPanelState } from '../composables/useParamPanelState'
import { useAppStore } from '../stores/app'
import { analyzeVectorWidth } from '../api/convert'
import type { WidthAnalysisResult } from '../types'
import ChannelSelector from './param/ChannelSelector.vue'
import ColorDBSelector from './param/ColorDBSelector.vue'
import ParamAdvancedSection from './param/ParamAdvancedSection.vue'
import ParamModeSwitch from './param/ParamModeSwitch.vue'
import ParamSimpleSection from './param/ParamSimpleSection.vue'
import WidthHistogram from './WidthHistogram.vue'

defineProps<{
  disabled?: boolean
}>()

const { t } = useI18n()

const {
  activeChannelPreset,
  applicablePresets,
  clusterMode,
  clusterModeOptions,
  availableChannels,
  clusterCountUpperBound,
  clusterCountValue,
  colorSpaceOptions,
  ditherOptions,
  error,
  filteredDBOptions,
  formatTooltip2Decimals,
  handleChannelKeysChange,
  imageDimensions,
  inlineLabelWidth,
  isAllChannelsSelected,
  isLuminaPreset,
  isRaster,
  isVector,
  loading,
  materialOptions,
  mode,
  modelPackAvailable,
  modelPackModeHint,
  modelValue,
  resetParams,
  selectedChannelKeys,
  selectedMaterial,
  selectedVendor,
  setClusterCount,
  simpleLabelWidth,
  simpleOutputInfo,
  supportsModelGate,
  targetDimensionUpperBound,
  targetHeightMm,
  targetWidthMm,
  tooltips,
  roundTo,
  update,
  vendorOptionsForMaterial,
  applyChannelPreset,
} = useParamPanelState()

const appStore = useAppStore()

const widthAnalysis = ref<WidthAnalysisResult | null>(null)
const widthAnalyzing = ref(false)

const nozzleDiameterMm = computed(() => {
  const ns = modelValue.value.nozzle_size
  return ns === 'n02' ? 0.2 : 0.4
})

const showColorLayersHint = computed(() => {
  const layers = modelValue.value.color_layers
  return layers !== undefined && layers !== 5 && layers !== 10
})

async function handleAnalyzeWidth() {
  const file = appStore.selectedFile
  if (!file || !isVector.value) return
  widthAnalyzing.value = true
  widthAnalysis.value = null
  try {
    widthAnalysis.value = await analyzeVectorWidth(file, {
      flip_y: modelValue.value.flip_y,
    })
  } catch (e) {
    console.error('Width analysis failed:', e)
  } finally {
    widthAnalyzing.value = false
  }
}

const isDoubleSided = computed(() => modelValue.value.double_sided ?? false)
const resolvedFaceOrientation = computed<'faceup' | 'facedown'>(() =>
  isDoubleSided.value ? 'facedown' : (modelValue.value.face_orientation ?? 'faceup'),
)
const faceOrientationBeforeDoubleSided = ref<'faceup' | 'facedown' | null>(null)

function handleFaceOrientationChange(v: string) {
  if (isDoubleSided.value) return
  update({ face_orientation: v as 'faceup' | 'facedown' })
}

function handleDoubleSidedChange(v: boolean) {
  if (v) {
    faceOrientationBeforeDoubleSided.value = modelValue.value.face_orientation ?? 'faceup'
    update({ double_sided: true, face_orientation: 'facedown' })
    return
  }
  const restoredFaceOrientation = faceOrientationBeforeDoubleSided.value ?? 'faceup'
  faceOrientationBeforeDoubleSided.value = null
  update({ double_sided: false, face_orientation: restoredFaceOrientation })
}

const canEnableTransparentLayer = computed(
  () => resolvedFaceOrientation.value === 'facedown' && !isDoubleSided.value,
)

watch(
  () => [isDoubleSided.value, modelValue.value.face_orientation] as const,
  ([doubleSided, faceOrientation]) => {
    if (!doubleSided) {
      faceOrientationBeforeDoubleSided.value = faceOrientation ?? 'faceup'
    }
    if (doubleSided && faceOrientation !== 'facedown') {
      update({ face_orientation: 'facedown' })
    }
  },
  { immediate: true },
)

watch(canEnableTransparentLayer, (can) => {
  if (!can && (modelValue.value.transparent_layer_mm ?? 0) > 0) {
    update({ transparent_layer_mm: 0 })
  }
})
</script>

<template>
  <NCard :title="t('param.title')" size="small">
    <template #header-extra>
      <NSpace :size="8" align="center">
        <NButton size="tiny" quaternary :disabled="loading" @click="resetParams">
          {{ t('common.reset') }}
        </NButton>
        <ParamModeSwitch v-model="mode" />
      </NSpace>
    </template>

    <NSpin :show="loading">
      <NAlert v-if="error" type="error" :title="error" style="margin-bottom: 12px" />

      <!-- ==================== SIMPLE MODE ==================== -->
      <ParamSimpleSection v-if="mode === 'simple'" :disabled="disabled || loading">
        <!-- Bambu preset selection -->
        <div class="param-inline-row">
          <NFormItem
            class="param-inline-item"
            label-placement="left"
            :label-width="inlineLabelWidth"
          >
            <template #label>
              <span class="tip-label">{{ t('param.nozzleSize') }}</span>
            </template>
            <NRadioGroup
              :value="modelValue.nozzle_size ?? 'n04'"
              size="small"
              @update:value="(v: string) => update({ nozzle_size: v as 'n02' | 'n04' })"
            >
              <NRadioButton value="n04">0.4mm</NRadioButton>
              <NRadioButton value="n02">0.2mm</NRadioButton>
            </NRadioGroup>
          </NFormItem>

          <NFormItem
            class="param-inline-item"
            label-placement="left"
            :label-width="inlineLabelWidth"
          >
            <template #label>
              <span class="tip-label">{{ t('param.faceDirection') }}</span>
            </template>
            <NRadioGroup
              :value="resolvedFaceOrientation"
              size="small"
              :disabled="isDoubleSided"
              @update:value="handleFaceOrientationChange"
            >
              <NRadioButton value="faceup">{{ t('param.faceUp') }}</NRadioButton>
              <NRadioButton value="facedown">{{ t('param.faceDown') }}</NRadioButton>
            </NRadioGroup>
          </NFormItem>
        </div>

        <div class="param-inline-row">
          <NFormItem
            class="param-inline-item"
            label-placement="left"
            :label-width="inlineLabelWidth"
          >
            <template #label>
              <NTooltip>
                <template #trigger>
                  <span class="tip-label">{{ t('param.baseLayers') }}</span>
                </template>
                {{ tooltips.base_layers }}
              </NTooltip>
            </template>
            <NInputNumber
              :value="modelValue.base_layers"
              :min="0"
              :step="1"
              :show-button="true"
              clearable
              @update:value="(v: number | null) => update({ base_layers: v ?? undefined })"
            />
          </NFormItem>

          <NFormItem
            class="param-inline-item"
            label-placement="left"
            :label-width="inlineLabelWidth"
          >
            <template #label>
              <NTooltip>
                <template #trigger>
                  <span class="tip-label">{{ t('param.doubleSided') }}</span>
                </template>
                {{ tooltips.double_sided }}
              </NTooltip>
            </template>
            <NSwitch
              :value="modelValue.double_sided ?? false"
              @update:value="handleDoubleSidedChange"
            />
          </NFormItem>

          <NFormItem
            v-if="canEnableTransparentLayer"
            label-placement="left"
            :label-width="simpleLabelWidth"
          >
            <template #label>
              <NTooltip>
                <template #trigger>
                  <span class="tip-label">{{ t('param.transparentLayer') }}</span>
                </template>
                {{ tooltips.transparent_layer_mm }}
              </NTooltip>
            </template>
            <NRadioGroup
              :value="modelValue.transparent_layer_mm ?? 0"
              @update:value="(v: number) => update({ transparent_layer_mm: v })"
            >
              <NRadioButton :value="0">{{ t('param.off') }}</NRadioButton>
              <NRadioButton :value="0.04">0.04mm</NRadioButton>
              <NRadioButton :value="0.08">0.08mm</NRadioButton>
            </NRadioGroup>
          </NFormItem>
        </div>

        <!-- Target width mm (shared) -->
        <NFormItem label-placement="left" :label-width="simpleLabelWidth">
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">{{ t('param.targetWidthMm') }}</span>
              </template>
              {{ tooltips.target_width_mm }}
            </NTooltip>
          </template>
          <div class="slider-input-row">
            <NSlider
              v-model:value="targetWidthMm"
              :min="1"
              :max="targetDimensionUpperBound"
              :step="1"
              class="slider-input-row__slider"
            />
            <NInputNumber
              v-model:value="targetWidthMm"
              :min="1"
              :max="targetDimensionUpperBound"
              :step="1"
              :show-button="false"
              class="slider-input-row__input number-input-right"
            />
          </div>
        </NFormItem>

        <!-- Target height mm (shared) -->
        <NFormItem label-placement="left" :label-width="simpleLabelWidth">
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">{{ t('param.targetHeightMm') }}</span>
              </template>
              {{ tooltips.target_height_mm }}
            </NTooltip>
          </template>
          <div class="slider-input-row">
            <NSlider
              v-model:value="targetHeightMm"
              :min="1"
              :max="targetDimensionUpperBound"
              :step="1"
              class="slider-input-row__slider"
            />
            <NInputNumber
              v-model:value="targetHeightMm"
              :min="1"
              :max="targetDimensionUpperBound"
              :step="1"
              :show-button="false"
              class="slider-input-row__input number-input-right"
            />
          </div>
        </NFormItem>

        <!-- Width analysis (vector only) -->
        <div v-if="isVector" class="width-analysis-section">
          <NButton
            :loading="widthAnalyzing"
            :disabled="!appStore.selectedFile || disabled"
            size="small"
            type="primary"
            ghost
            block
            @click="handleAnalyzeWidth"
          >
            {{ t('widthAnalysis.analyzeButton') }}
          </NButton>
          <NCollapse
            v-if="widthAnalysis && widthAnalysis.shapes.length > 0"
            :default-expanded-names="['width-result']"
          >
            <NCollapseItem :title="t('widthAnalysis.resultTitle')" name="width-result">
              <WidthHistogram
                :shapes="widthAnalysis.shapes"
                :nozzle-diameter="nozzleDiameterMm"
                :image-width-mm="widthAnalysis.image_width_mm"
                :image-height-mm="widthAnalysis.image_height_mm"
                :target-width-mm="modelValue.target_width_mm ?? 0"
                :target-height-mm="modelValue.target_height_mm ?? 0"
              />
            </NCollapseItem>
          </NCollapse>
        </div>

        <!-- Tessellation tolerance (vector only) -->
        <NFormItem v-if="isVector" label-placement="left" :label-width="simpleLabelWidth">
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">{{ t('param.tessellationTolerance') }}</span>
              </template>
              {{ tooltips.tessellation_tolerance_mm }}
            </NTooltip>
          </template>
          <NInputNumber
            :value="modelValue.tessellation_tolerance_mm ?? 0.03"
            :min="0.01"
            :max="1"
            :step="0.01"
            :precision="2"
            @update:value="
              (v: number | null) => update({ tessellation_tolerance_mm: roundTo(v ?? 0.03, 2) })
            "
          />
        </NFormItem>

        <NFormItem label-placement="left" :label-width="simpleLabelWidth">
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">{{ t('param.colorLayers') }}</span>
              </template>
              {{ tooltips.color_layers }}
            </NTooltip>
          </template>
          <NInputNumber
            :value="modelValue.color_layers"
            :min="1"
            :max="20"
            @update:value="(v: number | null) => update({ color_layers: v ?? 5 })"
          />
        </NFormItem>
        <NText v-if="showColorLayersHint" depth="3" style="font-size: 12px">
          {{ t('param.colorLayersHint') }}
        </NText>

        <ColorDBSelector
          :material="selectedMaterial"
          :vendor="selectedVendor"
          :material-options="materialOptions"
          :vendor-options="vendorOptionsForMaterial"
          :db-names="modelValue.db_names"
          :db-options="filteredDBOptions"
          :db-tooltip="tooltips.db_names"
          :is-lumina-preset="isLuminaPreset"
          @update:material="
            (v: string) => {
              selectedMaterial = v
            }
          "
          @update:vendor="
            (v: string) => {
              selectedVendor = v
            }
          "
          @update:db-names="(v: string[]) => update({ db_names: v })"
        />

        <!-- KMeans cluster count (raster only) -->
        <NFormItem v-if="isRaster" label-placement="left" :label-width="simpleLabelWidth">
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">{{ t('param.clusterCount') }}</span>
              </template>
              {{ tooltips.cluster_count }}
            </NTooltip>
          </template>
          <div class="slider-input-row">
            <NSelect
              v-model:value="clusterMode"
              :options="clusterModeOptions"
              size="small"
              style="width: 90px; flex-shrink: 0"
            />
            <template v-if="clusterMode === 'manual'">
              <NSlider
                v-model:value="clusterCountValue"
                :min="2"
                :max="clusterCountUpperBound"
                :step="1"
                class="slider-input-row__slider"
              />
              <NInputNumber
                :value="clusterCountValue"
                :min="2"
                :max="clusterCountUpperBound"
                :step="1"
                :show-button="false"
                class="slider-input-row__input number-input-right"
                @update:value="setClusterCount"
              />
            </template>
          </div>
        </NFormItem>

        <div v-if="isRaster" class="param-inline-row">
          <!-- Dither (raster only) -->
          <NFormItem
            class="param-inline-item"
            label-placement="left"
            :label-width="inlineLabelWidth"
          >
            <template #label>
              <NTooltip>
                <template #trigger>
                  <span class="tip-label">{{ t('param.halftone') }}</span>
                </template>
                {{ tooltips.dither }}
              </NTooltip>
            </template>
            <NSelect
              :value="modelValue.dither ?? 'none'"
              :options="ditherOptions"
              @update:value="(v: string) => update({ dither: v })"
            />
          </NFormItem>

          <!-- Model enable -->
          <NFormItem
            class="param-inline-item"
            label-placement="left"
            :label-width="inlineLabelWidth"
          >
            <template #label>
              <NTooltip>
                <template #trigger>
                  <span class="tip-label">{{ t('param.enableModel') }}</span>
                </template>
                {{ modelPackAvailable ? tooltips.model_enable : t('param.modelPackNoMatch') }}
              </NTooltip>
            </template>
            <NSwitch
              :value="modelValue.model_enable"
              :disabled="!modelPackAvailable"
              @update:value="(v: boolean) => update({ model_enable: v })"
            />
          </NFormItem>
          <NText v-if="modelPackModeHint && modelPackAvailable" depth="3" style="font-size: 12px">
            {{ modelPackModeHint }}
          </NText>
        </div>

        <NFormItem
          v-else-if="isVector && supportsModelGate"
          class="param-inline-item"
          label-placement="left"
          :label-width="simpleLabelWidth"
        >
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">{{ t('param.enableModel') }}</span>
              </template>
              {{ modelPackAvailable ? tooltips.model_enable : t('param.modelPackNoMatch') }}
            </NTooltip>
          </template>
          <NSwitch
            :value="modelValue.model_enable"
            :disabled="!modelPackAvailable"
            @update:value="(v: boolean) => update({ model_enable: v })"
          />
        </NFormItem>
        <NText
          v-if="modelPackModeHint && modelPackAvailable && isVector && supportsModelGate"
          depth="3"
          style="font-size: 12px"
        >
          {{ modelPackModeHint }}
        </NText>

        <NFormItem
          v-else-if="supportsModelGate"
          label-placement="left"
          :label-width="simpleLabelWidth"
        >
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">{{ t('param.enableModel') }}</span>
              </template>
              {{ modelPackAvailable ? tooltips.model_enable : t('param.modelPackNoMatch') }}
            </NTooltip>
          </template>
          <NSwitch
            :value="modelValue.model_enable"
            :disabled="!modelPackAvailable"
            @update:value="(v: boolean) => update({ model_enable: v })"
          />
        </NFormItem>
        <NText
          v-if="
            modelPackModeHint && modelPackAvailable && supportsModelGate && !isRaster && !isVector
          "
          depth="3"
          style="font-size: 12px"
        >
          {{ modelPackModeHint }}
        </NText>

        <NFormItem
          v-if="isRaster && modelValue.dither && modelValue.dither !== 'none'"
          label-placement="left"
          :label-width="simpleLabelWidth"
        >
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">{{ t('param.ditherStrength') }}</span>
              </template>
              {{ tooltips.dither_strength }}
            </NTooltip>
          </template>
          <NSlider
            :value="modelValue.dither_strength ?? 0.8"
            :min="0"
            :max="1"
            :step="0.05"
            :tooltip="true"
            :format-tooltip="formatTooltip2Decimals"
            @update:value="(v: number) => update({ dither_strength: roundTo(v, 2) })"
          />
        </NFormItem>

        <ChannelSelector
          collapse-name="channels"
          :tooltip-text="tooltips.allowed_channels"
          :selected-keys="selectedChannelKeys"
          :available-channels="availableChannels"
          :applicable-presets="applicablePresets"
          :active-preset="activeChannelPreset"
          :is-all-selected="isAllChannelsSelected"
          @update:selected-keys="handleChannelKeysChange"
          @apply:preset="applyChannelPreset"
        />

        <!-- Output info (raster only) -->
        <NAlert
          v-if="isRaster && simpleOutputInfo"
          type="info"
          :bordered="false"
          style="margin-top: 4px"
        >
          <div style="font-size: 12px; line-height: 1.6">
            <div>
              {{ t('param.outputPixels') }}
              <strong>{{ simpleOutputInfo.actualPxW }}×{{ simpleOutputInfo.actualPxH }}</strong>
              {{ t('param.pixels') }}
              <span v-if="imageDimensions"
                >({{ t('param.originalSize') }} {{ imageDimensions.width }}×{{
                  imageDimensions.height
                }})</span
              >
            </div>
            <div>
              {{ t('param.actualSize') }}
              <strong
                >{{ simpleOutputInfo.actualWidthMm }}×{{ simpleOutputInfo.actualHeightMm }}</strong
              >
              mm
            </div>
          </div>
        </NAlert>
        <NAlert
          v-if="isRaster && simpleOutputInfo?.upscaleRatio"
          type="warning"
          :bordered="false"
          style="margin-top: 4px"
        >
          <div style="font-size: 12px">
            {{ t('param.upscaleWarning', { ratio: simpleOutputInfo.upscaleRatio.toFixed(1) }) }}
          </div>
        </NAlert>
      </ParamSimpleSection>

      <!-- ==================== ADVANCED MODE ==================== -->
      <ParamAdvancedSection v-else :disabled="disabled || loading">
        <!-- Image processing group (raster has more items) -->
        <NCollapse default-expanded-names="imgproc" style="margin-bottom: 8px">
          <NCollapseItem
            :title="isRaster ? t('param.imageProcessing') : t('param.sizeSettings')"
            name="imgproc"
          >
            <NFormItem v-if="isRaster">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.scaleFactor') }}</span>
                  </template>
                  {{ tooltips.scale }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.scale"
                :min="0.01"
                :max="10"
                :step="0.1"
                :precision="1"
                @update:value="
                  (v: number | null) => update({ scale: v === null ? undefined : roundTo(v, 1) })
                "
              />
            </NFormItem>

            <NFormItem v-if="isRaster">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.maxWidthPx') }}</span>
                  </template>
                  {{ tooltips.max_width }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.max_width"
                :min="0"
                :max="4096"
                @update:value="(v: number | null) => update({ max_width: v ?? undefined })"
              />
            </NFormItem>

            <NFormItem v-if="isRaster">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.maxHeightPx') }}</span>
                  </template>
                  {{ tooltips.max_height }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.max_height"
                :min="0"
                :max="4096"
                @update:value="(v: number | null) => update({ max_height: v ?? undefined })"
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.targetWidthMm') }}</span>
                  </template>
                  {{ tooltips.target_width_mm }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.target_width_mm"
                :min="0"
                :max="targetDimensionUpperBound"
                :step="1"
                @update:value="(v: number | null) => update({ target_width_mm: v ?? 0 })"
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.targetHeightMm') }}</span>
                  </template>
                  {{ tooltips.target_height_mm }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.target_height_mm"
                :min="0"
                :max="targetDimensionUpperBound"
                :step="1"
                @update:value="(v: number | null) => update({ target_height_mm: v ?? 0 })"
              />
            </NFormItem>
          </NCollapseItem>
        </NCollapse>

        <!-- Vector parameters group (vector only) -->
        <NCollapse
          v-if="isVector"
          default-expanded-names="vector-params"
          style="margin-bottom: 8px"
        >
          <NCollapseItem :title="t('param.vectorParams')" name="vector-params">
            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.tessellationTolerance') }}</span>
                  </template>
                  {{ tooltips.tessellation_tolerance_mm }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.tessellation_tolerance_mm ?? 0.03"
                :min="0.01"
                :max="1"
                :step="0.01"
                :precision="2"
                @update:value="
                  (v: number | null) => update({ tessellation_tolerance_mm: roundTo(v ?? 0.03, 2) })
                "
              />
            </NFormItem>
          </NCollapseItem>
        </NCollapse>

        <!-- Bambu preset group -->
        <NCollapse default-expanded-names="bambu-preset" style="margin-bottom: 8px">
          <NCollapseItem :title="t('param.bambuPresets')" name="bambu-preset">
            <NFormItem>
              <template #label>
                <span class="tip-label">{{ t('param.nozzleSize') }}</span>
              </template>
              <NRadioGroup
                :value="modelValue.nozzle_size ?? 'n04'"
                size="small"
                @update:value="(v: string) => update({ nozzle_size: v as 'n02' | 'n04' })"
              >
                <NRadioButton value="n04">0.4mm</NRadioButton>
                <NRadioButton value="n02">0.2mm</NRadioButton>
              </NRadioGroup>
            </NFormItem>

            <NFormItem>
              <template #label>
                <span class="tip-label">{{ t('param.faceDirection') }}</span>
              </template>
              <NRadioGroup
                :value="resolvedFaceOrientation"
                size="small"
                :disabled="isDoubleSided"
                @update:value="handleFaceOrientationChange"
              >
                <NRadioButton value="faceup">{{ t('param.faceUp') }}</NRadioButton>
                <NRadioButton value="facedown">{{ t('param.faceDown') }}</NRadioButton>
              </NRadioGroup>
            </NFormItem>
          </NCollapseItem>
        </NCollapse>

        <!-- Geometry group -->
        <NCollapse default-expanded-names="geometry" style="margin-bottom: 8px">
          <NCollapseItem :title="t('param.geometryParams')" name="geometry">
            <NFormItem v-if="isRaster">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.pixelSizeMm') }}</span>
                  </template>
                  {{ tooltips.pixel_mm }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.pixel_mm"
                :min="0"
                :max="10"
                :step="0.01"
                :precision="2"
                @update:value="
                  (v: number | null) => update({ pixel_mm: v === null ? undefined : roundTo(v, 2) })
                "
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.layerHeightMm') }}</span>
                  </template>
                  {{ tooltips.layer_height_mm }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.layer_height_mm"
                :min="0"
                :max="1"
                :step="0.01"
                :precision="2"
                @update:value="
                  (v: number | null) =>
                    update({ layer_height_mm: v === null ? undefined : roundTo(v, 2) })
                "
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.baseLayers') }}</span>
                  </template>
                  {{ tooltips.base_layers }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.base_layers"
                :min="0"
                :step="1"
                :show-button="true"
                clearable
                @update:value="(v: number | null) => update({ base_layers: v ?? undefined })"
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.doubleSided') }}</span>
                  </template>
                  {{ tooltips.double_sided }}
                </NTooltip>
              </template>
              <NSwitch
                :value="modelValue.double_sided ?? false"
                @update:value="handleDoubleSidedChange"
              />
            </NFormItem>

            <NFormItem v-if="canEnableTransparentLayer">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.transparentLayer') }}</span>
                  </template>
                  {{ tooltips.transparent_layer_mm }}
                </NTooltip>
              </template>
              <NRadioGroup
                :value="modelValue.transparent_layer_mm ?? 0"
                @update:value="(v: number) => update({ transparent_layer_mm: v })"
              >
                <NRadioButton :value="0">{{ t('param.off') }}</NRadioButton>
                <NRadioButton :value="0.04">0.04mm</NRadioButton>
                <NRadioButton :value="0.08">0.08mm</NRadioButton>
              </NRadioGroup>
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.flipY') }}</span>
                  </template>
                  {{ tooltips.flip_y }}
                </NTooltip>
              </template>
              <NSwitch
                :value="modelValue.flip_y"
                @update:value="(v: boolean) => update({ flip_y: v })"
              />
            </NFormItem>
          </NCollapseItem>
        </NCollapse>

        <!-- Color matching group -->
        <NCollapse default-expanded-names="matching" style="margin-bottom: 8px">
          <NCollapseItem :title="t('param.colorMatching')" name="matching">
            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.colorLayers') }}</span>
                  </template>
                  {{ tooltips.color_layers }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.color_layers"
                :min="1"
                :max="20"
                @update:value="(v: number | null) => update({ color_layers: v ?? 5 })"
              />
            </NFormItem>
            <NText v-if="showColorLayersHint" depth="3" style="font-size: 12px">
              {{ t('param.colorLayersHint') }}
            </NText>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.colorSpace') }}</span>
                  </template>
                  {{ tooltips.color_space }}
                </NTooltip>
              </template>
              <NSelect
                :value="modelValue.color_space"
                :options="colorSpaceOptions"
                @update:value="(v: string) => update({ color_space: v })"
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.kCandidates') }}</span>
                  </template>
                  {{ tooltips.k_candidates }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.k_candidates"
                :min="1"
                :max="64"
                @update:value="(v: number | null) => update({ k_candidates: v ?? undefined })"
              />
            </NFormItem>

            <NFormItem v-if="isRaster">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.clusterCount') }}</span>
                  </template>
                  {{ tooltips.cluster_count }}
                </NTooltip>
              </template>
              <NSpace align="center">
                <NSelect
                  v-model:value="clusterMode"
                  :options="clusterModeOptions"
                  size="small"
                  style="width: 90px"
                />
                <NInputNumber
                  v-if="clusterMode === 'manual'"
                  :value="clusterCountValue"
                  :min="2"
                  :max="clusterCountUpperBound"
                  @update:value="setClusterCount"
                />
              </NSpace>
            </NFormItem>

            <NFormItem v-if="isRaster">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.halftone') }}</span>
                  </template>
                  {{ tooltips.dither }}
                </NTooltip>
              </template>
              <NSelect
                :value="modelValue.dither ?? 'none'"
                :options="ditherOptions"
                @update:value="(v: string) => update({ dither: v })"
              />
            </NFormItem>

            <NFormItem v-if="isRaster && modelValue.dither && modelValue.dither !== 'none'">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.ditherStrength') }}</span>
                  </template>
                  {{ tooltips.dither_strength }}
                </NTooltip>
              </template>
              <NSlider
                :value="modelValue.dither_strength ?? 0.8"
                :min="0"
                :max="1"
                :step="0.05"
                :tooltip="true"
                :format-tooltip="formatTooltip2Decimals"
                @update:value="(v: number) => update({ dither_strength: roundTo(v, 2) })"
              />
            </NFormItem>

            <ColorDBSelector
              :material="selectedMaterial"
              :vendor="selectedVendor"
              :material-options="materialOptions"
              :vendor-options="vendorOptionsForMaterial"
              :db-names="modelValue.db_names"
              :db-options="filteredDBOptions"
              :db-tooltip="tooltips.db_names"
              :is-lumina-preset="isLuminaPreset"
              @update:material="
                (v: string) => {
                  selectedMaterial = v
                }
              "
              @update:vendor="
                (v: string) => {
                  selectedVendor = v
                }
              "
              @update:db-names="(v: string[]) => update({ db_names: v })"
            />

            <ChannelSelector
              collapse-name="channels-adv"
              :tooltip-text="tooltips.allowed_channels"
              :selected-keys="selectedChannelKeys"
              :available-channels="availableChannels"
              :applicable-presets="applicablePresets"
              :active-preset="activeChannelPreset"
              :is-all-selected="isAllChannelsSelected"
              @update:selected-keys="handleChannelKeysChange"
              @apply:preset="applyChannelPreset"
            />
          </NCollapseItem>
        </NCollapse>

        <!-- Model gate group -->
        <NCollapse v-if="supportsModelGate" style="margin-bottom: 8px">
          <NCollapseItem :title="t('param.modelGating')" name="model-gate">
            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.enableModel') }}</span>
                  </template>
                  {{ modelPackAvailable ? tooltips.model_enable : t('param.modelPackNoMatch') }}
                </NTooltip>
              </template>
              <NSwitch
                :value="modelValue.model_enable"
                :disabled="!modelPackAvailable"
                @update:value="(v: boolean) => update({ model_enable: v })"
              />
            </NFormItem>
            <NText
              v-if="modelPackModeHint && modelPackAvailable"
              depth="3"
              style="font-size: 12px; margin-bottom: 8px"
            >
              {{ modelPackModeHint }}
            </NText>

            <NFormItem v-if="modelValue.model_enable">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.modelOnly') }}</span>
                  </template>
                  {{ tooltips.model_only }}
                </NTooltip>
              </template>
              <NSwitch
                :value="modelValue.model_only"
                @update:value="(v: boolean) => update({ model_only: v })"
              />
            </NFormItem>

            <NFormItem v-if="modelValue.model_enable">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.modelThreshold') }}</span>
                  </template>
                  {{ tooltips.model_threshold }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.model_threshold"
                :step="0.5"
                :precision="1"
                @update:value="
                  (v: number | null) =>
                    update({ model_threshold: v === null ? undefined : roundTo(v, 1) })
                "
              />
            </NFormItem>

            <NFormItem v-if="modelValue.model_enable">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.modelMargin') }}</span>
                  </template>
                  {{ tooltips.model_margin }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.model_margin"
                :step="0.5"
                :precision="1"
                @update:value="
                  (v: number | null) =>
                    update({ model_margin: v === null ? undefined : roundTo(v, 1) })
                "
              />
            </NFormItem>
          </NCollapseItem>
        </NCollapse>

        <!-- Output options group -->
        <NCollapse>
          <NCollapseItem :title="t('param.outputOptions')" name="output">
            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.generatePreview') }}</span>
                  </template>
                  {{ tooltips.generate_preview }}
                </NTooltip>
              </template>
              <NSwitch
                :value="modelValue.generate_preview"
                @update:value="(v: boolean) => update({ generate_preview: v })"
              />
            </NFormItem>

            <NFormItem v-if="isRaster">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">{{ t('param.generateSourceMask') }}</span>
                  </template>
                  {{ tooltips.generate_source_mask }}
                </NTooltip>
              </template>
              <NSwitch
                :value="modelValue.generate_source_mask"
                @update:value="(v: boolean) => update({ generate_source_mask: v })"
              />
            </NFormItem>
          </NCollapseItem>
        </NCollapse>
      </ParamAdvancedSection>
    </NSpin>
  </NCard>
</template>

<style scoped>
.tip-label {
  cursor: help;
  border-bottom: 1px dashed var(--n-text-color-3);
}

.param-inline-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  column-gap: 8px;
}

.param-inline-row--single {
  grid-template-columns: 1fr;
}

.param-inline-item {
  min-width: 0;
}

.slider-input-row {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}

.slider-input-row__slider {
  flex: 1;
  min-width: 0;
}

.slider-input-row__input {
  width: 104px;
}

.pixel-size-row {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}

.pixel-size-row__select {
  flex: 1;
  min-width: 0;
}

.pixel-size-row__input {
  width: 112px;
}

.number-input-right :deep(.n-input__input-el) {
  text-align: right;
}

.width-analysis-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin: 4px 0 8px;
}

@media (max-width: 920px) {
  .param-inline-row {
    grid-template-columns: 1fr;
  }

  .slider-input-row__input {
    width: 96px;
  }

  .pixel-size-row__input {
    width: 100px;
  }
}
</style>
