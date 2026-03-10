<script setup lang="ts">
import { NAlert, NFormItem, NSelect, NTooltip } from 'naive-ui'
import type { SelectOption } from 'naive-ui'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps<{
  material: string
  vendor: string
  materialOptions: SelectOption[]
  vendorOptions: SelectOption[]
  dbNames?: string[]
  dbOptions: SelectOption[]
  dbTooltip: string
  isLuminaPreset: boolean
}>()

const emit = defineEmits<{
  'update:material': [value: string]
  'update:vendor': [value: string]
  'update:dbNames': [value: string[]]
}>()
</script>

<template>
  <div class="selector-inline-row">
    <NFormItem
      :label="t('colordb.materialType')"
      class="selector-inline-item"
      label-placement="left"
      :label-width="96"
    >
      <NSelect
        size="small"
        :value="props.material"
        :options="props.materialOptions"
        @update:value="(v: string) => emit('update:material', v)"
      />
    </NFormItem>

    <NFormItem
      :label="t('colordb.vendor')"
      class="selector-inline-item"
      label-placement="left"
      :label-width="96"
    >
      <NSelect
        size="small"
        :value="props.vendor"
        :options="props.vendorOptions"
        @update:value="(v: string) => emit('update:vendor', v)"
      />
    </NFormItem>
  </div>

  <NFormItem label-placement="left" :label-width="96">
    <template #label>
      <NTooltip>
        <template #trigger>
          <span class="tip-label">{{ t('colordb.database') }}</span>
        </template>
        {{ props.dbTooltip }}
      </NTooltip>
    </template>
    <NSelect
      size="small"
      :value="props.dbNames"
      :options="props.dbOptions"
      multiple
      :max-tag-count="1"
      :placeholder="t('colordb.selectDatabase')"
      @update:value="(v: string[]) => emit('update:dbNames', v)"
    />
  </NFormItem>

  <NAlert
    v-if="props.isLuminaPreset"
    type="info"
    :bordered="false"
    style="margin-bottom: 12px; font-size: 12px"
  >
    {{ t('colordb.presetFrom') }}
    <a href="https://github.com/MOVIBALE/Lumina-Layers" target="_blank" rel="noopener"
      >Lumina-Layers</a
    >
    {{ t('colordb.openSource') }}
  </NAlert>
</template>

<style scoped>
.tip-label {
  cursor: help;
  border-bottom: 1px dashed var(--n-text-color-3);
}

.selector-inline-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  column-gap: 8px;
}

.selector-inline-item {
  min-width: 0;
  margin-bottom: 8px;
}

@media (max-width: 900px) {
  .selector-inline-row {
    grid-template-columns: 1fr;
  }
}
</style>
