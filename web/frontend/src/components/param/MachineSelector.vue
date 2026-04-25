<template>
  <NFormItem
    v-if="enabled"
    label-placement="left"
    :label-width="labelWidth"
    class="machine-selector"
  >
    <template #label>
      <NTooltip>
        <template #trigger>
          <span class="tip-label">{{ t('param.targetMachine') }}</span>
        </template>
        {{ t('param.targetMachineTooltip') }}
      </NTooltip>
    </template>
    <NSelect
      :value="selectedValue"
      :options="machineOptions"
      :placeholder="t('param.targetMachineHint')"
      :consistent-menu-width="false"
      size="small"
      @update:value="onChange"
    />
  </NFormItem>
</template>

<script setup lang="ts">
import { NFormItem, NSelect, NTooltip } from 'naive-ui'
import { computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useAppStore } from '../../stores/app'
import type { NozzleSize } from '../../types'

const props = defineProps<{
  /// Currently selected target_machine name (empty string -> catalog default).
  modelValue: string | undefined
  /// Optional nozzle size: when provided, hides machines that don't expose this nozzle.
  nozzleSize?: NozzleSize
  /// Form item label-width override (defaults to 90px).
  labelWidth?: number | string
}>()

const emit = defineEmits<{
  (event: 'update:modelValue', value: string): void
}>()

const { t } = useI18n()
const appStore = useAppStore()

const labelWidth = computed(() => props.labelWidth ?? 90)

onMounted(() => {
  void appStore.ensureMachines()
})

const requestedNozzleStr = computed(() => {
  if (!props.nozzleSize) return undefined
  return props.nozzleSize === 'n02' ? '0.2' : '0.4'
})

const filteredMachines = computed(() => {
  if (!requestedNozzleStr.value) return appStore.machines
  return appStore.machines.filter((m) => m.nozzles.some((n) => n === requestedNozzleStr.value))
})

const enabled = computed(() => filteredMachines.value.length > 0)

const machineOptions = computed(() =>
  filteredMachines.value.map((m) => ({
    label: m.name,
    value: m.name,
  })),
)

const selectedValue = computed(() => {
  const current = props.modelValue ?? ''
  if (current && filteredMachines.value.some((m) => m.name === current)) {
    return current
  }
  if (
    appStore.defaultMachine &&
    filteredMachines.value.some((m) => m.name === appStore.defaultMachine)
  ) {
    return appStore.defaultMachine
  }
  return filteredMachines.value[0]?.name ?? ''
})

function onChange(value: string) {
  emit('update:modelValue', value)
}
</script>

<style scoped>
.machine-selector {
  min-width: 200px;
}
</style>
