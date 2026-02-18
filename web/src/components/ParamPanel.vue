<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import {
  NCard,
  NForm,
  NFormItem,
  NSelect,
  NInputNumber,
  NSwitch,
  NCollapse,
  NCollapseItem,
  NSpin,
  NAlert,
  NTooltip,
  NCheckbox,
  NCheckboxGroup,
  NSpace,
} from 'naive-ui'
import type { SelectOption } from 'naive-ui'
import { fetchDefaults, fetchColorDBs } from '../api'
import type { ConvertParams, ColorDBInfo, DefaultConfig, PaletteChannel } from '../types'

const props = defineProps<{
  modelValue: ConvertParams
  disabled?: boolean
  refreshTrigger?: number
}>()

const emit = defineEmits<{
  'update:modelValue': [params: ConvertParams]
}>()

const loading = ref(true)
const error = ref<string | null>(null)
const colorDBs = ref<ColorDBInfo[]>([])
const defaults = ref<DefaultConfig | null>(null)

const dbOptions = ref<SelectOption[]>([])

const printModeOptions: SelectOption[] = [
  { label: '0.08mm × 5 层', value: '0.08x5' },
  { label: '0.04mm × 10 层', value: '0.04x10' },
]

const colorSpaceOptions: SelectOption[] = [
  { label: 'Lab', value: 'lab' },
  { label: 'RGB', value: 'rgb' },
]

const tooltips: Record<string, string> = {
  print_mode:
    '决定色层数和层高。0.08mm x 5 层: 层高 0.08mm，5 层叠色；0.04mm x 10 层: 层高 0.04mm，10 层叠色。两者总叠色厚度相同 (0.4mm)',
  color_space:
    '颜色匹配时使用的色彩空间。Lab 更符合人眼感知（推荐）；RGB 为直接 RGB 距离计算',
  max_width:
    '图像缩放后的最大宽度（像素），超出时等比缩小。值越小处理越快，输出模型越小',
  max_height:
    '图像缩放后的最大高度（像素），超出时等比缩小。值越小处理越快，输出模型越小',
  cluster_count:
    '对图像像素进行 K-Means 聚类后再匹配颜色。值越大颜色越精细，0 或 1 表示不聚类（逐像素匹配）。适合对像素画、动漫等色块较大的图像启用。需要根据图像挑选合适参数',
  db_names:
    '用于颜色匹配的 ColorDB，支持多选。匹配时会在所有选中的数据库中寻找最佳配方',
  allowed_channels:
    '选择参与配方生成的颜色通道。未选中的通道将被排除，使用这些通道的配方条目也会被自动过滤。默认使用全部通道',
  scale:
    '在最大宽高限制之前先对图像进行等比缩放，1.0 表示不缩放',
  k_candidates:
    '颜色匹配时从 KD-Tree 中取 K 个最近邻候选，再从中选择最优。值越大匹配越精确但越慢，1 表示直接取最近邻',
  flip_y:
    '构建 3D 模型时翻转 Y 轴方向，开启时图像顶部对应模型顶部',
  pixel_mm:
    '每个像素对应的实际尺寸（毫米），决定输出模型的物理大小。0 表示自动从 ColorDB 配置推导（通常为线宽）',
  layer_height_mm:
    '3D 打印的层高（毫米），决定模型 Z 方向分辨率。0 表示自动从打印模式推导',
  model_enable:
    '是否启用神经网络模型辅助匹配。开启后，当 ColorDB 匹配质量不佳时会尝试用模型预测更优配方',
  model_only:
    '跳过 ColorDB 匹配，完全使用模型预测配方。需要加载模型包',
  model_threshold:
    '色差 (DeltaE) 阈值，仅当 ColorDB 匹配色差超过此值时才启用模型。负值使用模型包默认值',
  model_margin:
    '模型结果需优于 ColorDB 结果至少此色差值才会被采用，防止模型轻微改善时替换稳定的 DB 结果。负值使用模型包默认值',
  generate_preview:
    '生成匹配后的颜色预览图（PNG），展示每个像素最终匹配到的颜色',
  generate_source_mask:
    '生成来源掩码图（PNG），白色像素表示使用了模型预测的配方，黑色表示使用了 ColorDB 匹配',
}

function update(partial: Partial<ConvertParams>) {
  emit('update:modelValue', { ...props.modelValue, ...partial })
}

// --- Channel filtering ---

function channelKey(ch: PaletteChannel): string {
  return `${ch.color}|${ch.material}`
}

interface ChannelOption {
  key: string
  color: string
  material: string
}

const selectedChannelKeys = ref<string[]>([])

const availableChannels = computed<ChannelOption[]>(() => {
  const selectedNames = props.modelValue.db_names ?? []
  const seen = new Set<string>()
  const channels: ChannelOption[] = []
  for (const db of colorDBs.value) {
    if (!selectedNames.includes(db.name)) continue
    for (const ch of db.palette) {
      const k = channelKey(ch)
      if (!seen.has(k)) {
        seen.add(k)
        channels.push({ key: k, color: ch.color, material: ch.material })
      }
    }
  }
  return channels
})

const isAllChannelsSelected = computed(() => {
  return (
    availableChannels.value.length > 0 &&
    selectedChannelKeys.value.length === availableChannels.value.length
  )
})

const isIndeterminate = computed(() => {
  return (
    selectedChannelKeys.value.length > 0 &&
    selectedChannelKeys.value.length < availableChannels.value.length
  )
})

function emitChannelUpdate() {
  const allKeys = new Set(availableChannels.value.map((c) => c.key))
  const selectedSet = new Set(selectedChannelKeys.value)
  if (
    selectedChannelKeys.value.length >= availableChannels.value.length &&
    [...allKeys].every((k) => selectedSet.has(k))
  ) {
    update({ allowed_channels: undefined })
  } else {
    const channels = availableChannels.value
      .filter((c) => selectedSet.has(c.key))
      .map((c) => ({ color: c.color, material: c.material }))
    update({ allowed_channels: channels })
  }
}

function handleChannelKeysChange(keys: (string | number)[]) {
  selectedChannelKeys.value = keys.map(String)
  emitChannelUpdate()
}

function toggleAllChannels(checked: boolean) {
  if (checked) {
    selectedChannelKeys.value = availableChannels.value.map((c) => c.key)
  } else {
    selectedChannelKeys.value = []
  }
  emitChannelUpdate()
}

watch(
  availableChannels,
  (newChannels, oldChannels) => {
    const oldKeys = new Set((oldChannels ?? []).map((c) => c.key))
    const newKeys = new Set(newChannels.map((c) => c.key))

    // 可用通道集合未实际变化时跳过，避免因对象引用不同导致的无限循环
    if (
      oldKeys.size === newKeys.size &&
      [...oldKeys].every((k) => newKeys.has(k))
    ) {
      return
    }

    let updated = selectedChannelKeys.value.filter((k) => newKeys.has(k))

    for (const ch of newChannels) {
      if (!oldKeys.has(ch.key) && !updated.includes(ch.key)) {
        updated.push(ch.key)
      }
    }

    if (updated.length === 0 && newChannels.length > 0) {
      updated = newChannels.map((c) => c.key)
    }

    selectedChannelKeys.value = updated
    emitChannelUpdate()
  },
)

function buildDBOptions(dbs: ColorDBInfo[]): SelectOption[] {
  return dbs.map((db) => {
    const suffix = db.source === 'session' ? ' (自定义)' : ''
    return {
      label: `${db.name} (${db.num_entries} 色, ${db.num_channels} 通道)${suffix}`,
      value: db.name,
    }
  })
}

async function loadColorDBs() {
  try {
    const dbsData = await fetchColorDBs()
    colorDBs.value = dbsData
    dbOptions.value = buildDBOptions(dbsData)
  } catch {
    // Keep existing options on refresh failure
  }
}

watch(
  () => props.refreshTrigger,
  () => { loadColorDBs() },
)

onMounted(async () => {
  try {
    const [defaultsData, dbsData] = await Promise.all([fetchDefaults(), fetchColorDBs()])
    defaults.value = defaultsData
    colorDBs.value = dbsData
    dbOptions.value = buildDBOptions(dbsData)

    const allDbNames = dbsData.filter((db) => db.source !== 'session').map((db) => db.name)
    emit('update:modelValue', {
      print_mode: defaultsData.print_mode,
      color_space: defaultsData.color_space,
      max_width: defaultsData.max_width,
      max_height: defaultsData.max_height,
      scale: defaultsData.scale,
      k_candidates: defaultsData.k_candidates,
      cluster_count: defaultsData.cluster_count,
      model_enable: defaultsData.model_enable,
      model_only: defaultsData.model_only,
      model_threshold: defaultsData.model_threshold,
      model_margin: defaultsData.model_margin,
      flip_y: defaultsData.flip_y,
      pixel_mm: defaultsData.pixel_mm,
      layer_height_mm: defaultsData.layer_height_mm,
      generate_preview: defaultsData.generate_preview,
      generate_source_mask: defaultsData.generate_source_mask,
      db_names: allDbNames,
    })
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : '加载配置失败'
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <NCard title="参数配置" size="small">
    <NSpin :show="loading">
      <NAlert v-if="error" type="error" :title="error" style="margin-bottom: 12px" />

      <NForm
        label-placement="left"
        label-width="auto"
        :disabled="disabled || loading"
        size="small"
      >
        <!-- Common parameters -->
        <NFormItem>
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">打印模式</span>
              </template>
              {{ tooltips.print_mode }}
            </NTooltip>
          </template>
          <NSelect
            :value="modelValue.print_mode"
            :options="printModeOptions"
            @update:value="(v: string) => update({ print_mode: v })"
          />
        </NFormItem>

        <NFormItem>
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">色彩空间</span>
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
                <span class="tip-label">最大宽度</span>
              </template>
              {{ tooltips.max_width }}
            </NTooltip>
          </template>
          <NInputNumber
            :value="modelValue.max_width"
            :min="1"
            :max="4096"
            @update:value="(v: number | null) => update({ max_width: v ?? undefined })"
          />
        </NFormItem>

        <NFormItem>
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">最大高度</span>
              </template>
              {{ tooltips.max_height }}
            </NTooltip>
          </template>
          <NInputNumber
            :value="modelValue.max_height"
            :min="1"
            :max="4096"
            @update:value="(v: number | null) => update({ max_height: v ?? undefined })"
          />
        </NFormItem>

        <NFormItem>
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">聚类数</span>
              </template>
              {{ tooltips.cluster_count }}
            </NTooltip>
          </template>
          <NInputNumber
            :value="modelValue.cluster_count"
            :min="0"
            :max="65536"
            @update:value="(v: number | null) => update({ cluster_count: v ?? undefined })"
          />
        </NFormItem>

        <NFormItem>
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">启用模型</span>
              </template>
              {{ tooltips.model_enable }}
            </NTooltip>
          </template>
          <NSwitch
            :value="modelValue.model_enable"
            @update:value="(v: boolean) => update({ model_enable: v })"
          />
        </NFormItem>

        <NFormItem>
          <template #label>
            <NTooltip>
              <template #trigger>
                <span class="tip-label">颜色数据库</span>
              </template>
              {{ tooltips.db_names }}
            </NTooltip>
          </template>
          <NSelect
            :value="modelValue.db_names"
            :options="dbOptions"
            multiple
            placeholder="选择颜色数据库"
            @update:value="(v: string[]) => update({ db_names: v })"
          />
        </NFormItem>

        <NCollapse v-if="availableChannels.length > 0" style="margin-bottom: 12px">
          <NCollapseItem name="channels">
            <template #header>
              <NTooltip>
                <template #trigger>
                  <span class="tip-label">颜色通道筛选</span>
                </template>
                {{ tooltips.allowed_channels }}
              </NTooltip>
            </template>
            <template #header-extra>
              <span style="font-size: 12px; color: var(--n-text-color-3)">
                {{ isAllChannelsSelected ? '全部' : `${selectedChannelKeys.length}/${availableChannels.length}` }}
              </span>
            </template>
            <div style="width: 100%">
              <NCheckbox
                :checked="isAllChannelsSelected"
                :indeterminate="isIndeterminate"
                style="margin-bottom: 6px"
                @update:checked="toggleAllChannels"
              >
                全选
              </NCheckbox>
              <NCheckboxGroup
                :value="selectedChannelKeys"
                @update:value="handleChannelKeysChange"
              >
                <NSpace item-style="display: flex">
                  <NCheckbox
                    v-for="ch in availableChannels"
                    :key="ch.key"
                    :value="ch.key"
                    :label="`${ch.color} (${ch.material})`"
                  />
                </NSpace>
              </NCheckboxGroup>
            </div>
          </NCollapseItem>
        </NCollapse>



        <!-- Advanced parameters -->
        <NCollapse>
          <NCollapseItem title="高级参数" name="advanced">
            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">缩放倍率</span>
                  </template>
                  {{ tooltips.scale }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.scale"
                :min="0.01"
                :max="10"
                :step="0.1"
                @update:value="(v: number | null) => update({ scale: v ?? undefined })"
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">K 候选数</span>
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

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">垂直翻转</span>
                  </template>
                  {{ tooltips.flip_y }}
                </NTooltip>
              </template>
              <NSwitch
                :value="modelValue.flip_y"
                @update:value="(v: boolean) => update({ flip_y: v })"
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">像素尺寸 (mm)</span>
                  </template>
                  {{ tooltips.pixel_mm }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.pixel_mm"
                :min="0"
                :max="10"
                :step="0.01"
                @update:value="(v: number | null) => update({ pixel_mm: v ?? undefined })"
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">层高 (mm)</span>
                  </template>
                  {{ tooltips.layer_height_mm }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.layer_height_mm"
                :min="0"
                :max="1"
                :step="0.01"
                @update:value="(v: number | null) => update({ layer_height_mm: v ?? undefined })"
              />
            </NFormItem>

            <NFormItem v-if="modelValue.model_enable">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">仅使用模型</span>
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
                    <span class="tip-label">模型阈值</span>
                  </template>
                  {{ tooltips.model_threshold }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.model_threshold"
                :step="0.5"
                @update:value="(v: number | null) => update({ model_threshold: v ?? undefined })"
              />
            </NFormItem>

            <NFormItem v-if="modelValue.model_enable">
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">模型边距</span>
                  </template>
                  {{ tooltips.model_margin }}
                </NTooltip>
              </template>
              <NInputNumber
                :value="modelValue.model_margin"
                :step="0.5"
                @update:value="(v: number | null) => update({ model_margin: v ?? undefined })"
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">生成预览图</span>
                  </template>
                  {{ tooltips.generate_preview }}
                </NTooltip>
              </template>
              <NSwitch
                :value="modelValue.generate_preview"
                @update:value="(v: boolean) => update({ generate_preview: v })"
              />
            </NFormItem>

            <NFormItem>
              <template #label>
                <NTooltip>
                  <template #trigger>
                    <span class="tip-label">生成源掩码</span>
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
      </NForm>
    </NSpin>
  </NCard>
</template>

<style scoped>
.tip-label {
  cursor: help;
  border-bottom: 1px dashed #999;
}
</style>
