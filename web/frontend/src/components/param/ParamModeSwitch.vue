<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { NRadioGroup, NRadioButton } from 'naive-ui'

const { t } = useI18n()

const props = defineProps<{
  modelValue: 'simple' | 'advanced'
}>()

const emit = defineEmits<{
  'update:modelValue': [value: 'simple' | 'advanced']
}>()
</script>

<template>
  <NRadioGroup
    :value="props.modelValue"
    size="small"
    @update:value="(v: 'simple' | 'advanced') => emit('update:modelValue', v)"
  >
    <NRadioButton value="simple">{{ t('param.mode.simple') }}</NRadioButton>
    <NRadioButton value="advanced">{{ t('param.mode.advanced') }}</NRadioButton>
  </NRadioGroup>
</template>
