<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import {
  NCollapse,
  NCollapseItem,
  NRadioGroup,
  NRadioButton,
  NCheckboxGroup,
  NCheckbox,
  NSpace,
  NTooltip,
} from 'naive-ui'

const { t } = useI18n()

type ChannelItem = {
  key: string
  color: string
  material: string
}

type ChannelPreset = {
  label: string
  value: string
}

const props = defineProps<{
  collapseName: string
  tooltipText: string
  selectedKeys: string[]
  availableChannels: ChannelItem[]
  applicablePresets: ChannelPreset[]
  activePreset: string | null
  isAllSelected: boolean
}>()

const emit = defineEmits<{
  'update:selectedKeys': [keys: string[]]
  'apply:preset': [value: string]
}>()

function handleChannelKeysChange(keys: Array<string | number>) {
  emit('update:selectedKeys', keys.map(String))
}
</script>

<template>
  <NCollapse v-if="props.availableChannels.length > 0" style="margin-bottom: 12px">
    <NCollapseItem :name="props.collapseName">
      <template #header>
        <NTooltip>
          <template #trigger>
            <span class="tip-label">{{ t('param.channelFilter') }}</span>
          </template>
          {{ props.tooltipText }}
        </NTooltip>
      </template>
      <template #header-extra>
        <span style="font-size: 12px; color: var(--n-text-color-3)">
          {{
            props.isAllSelected
              ? t('common.all')
              : `${props.selectedKeys.length}/${props.availableChannels.length}`
          }}
        </span>
      </template>
      <div style="width: 100%">
        <NRadioGroup
          v-if="props.applicablePresets.length > 1"
          :value="props.activePreset"
          size="small"
          style="margin-bottom: 8px"
          @update:value="(v: string) => emit('apply:preset', v)"
        >
          <NRadioButton v-for="p in props.applicablePresets" :key="p.value" :value="p.value">
            {{ p.label }}
          </NRadioButton>
        </NRadioGroup>
        <NCheckboxGroup :value="props.selectedKeys" @update:value="handleChannelKeysChange">
          <NSpace item-style="display: flex">
            <NCheckbox
              v-for="ch in props.availableChannels"
              :key="ch.key"
              :value="ch.key"
              :label="`${ch.color} (${ch.material})`"
            />
          </NSpace>
        </NCheckboxGroup>
      </div>
    </NCollapseItem>
  </NCollapse>
</template>

<style scoped>
.tip-label {
  cursor: help;
  border-bottom: 1px dashed var(--n-text-color-3);
}
</style>
