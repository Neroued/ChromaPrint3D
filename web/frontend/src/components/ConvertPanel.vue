<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { NButton, NCard, NText, NSpace, NAlert } from 'naive-ui'
import { useI18n } from 'vue-i18n'
import { useAsyncTask } from '../composables/useAsyncTask'
import { useBlobDownload } from '../composables/useBlobDownload'
import {
  fetchConvertTaskStatus,
  submitConvertTask,
  submitMatchOnlyTask,
} from '../services/convertService'
import { getResultPath } from '../services/resultService'
import { mapTaskError } from '../domain/error/taskErrorMapping'
import { useAppStore } from '../stores/app'
import ProgressActionGroup from './common/ProgressActionGroup.vue'
import type { TaskStatus } from '../types'

const appStore = useAppStore()
const { selectedFile, params, inputType, completedTask } = storeToRefs(appStore)
const { downloadByUrl } = useBlobDownload()
const { t } = useI18n()

function getStageLabel(stage: string): string {
  const key = `convert.stages.${stage}`
  const result = t(key)
  return result !== key ? result : stage
}

interface StageWeight {
  start: number
  weight: number
}

const stageWeights: Record<string, StageWeight> = {
  loading_resources: { start: 0.0, weight: 0.05 },
  preprocessing: { start: 0.05, weight: 0.3 },
  matching: { start: 0.35, weight: 0.15 },
  building_model: { start: 0.5, weight: 0.2 },
  exporting: { start: 0.7, weight: 0.3 },
}

function computeOverallProgress(stage: string, stageProgress: number): number {
  const w = stageWeights[stage]
  if (!w) return 0
  return Math.min(1, w.start + w.weight * Math.max(0, Math.min(1, stageProgress)))
}

function submitCurrentTask() {
  const file = selectedFile.value!
  return submitConvertTask(file, params.value, inputType.value)
}

const {
  status: taskStatus,
  loading,
  error,
  isCompleted,
  submit: submitTask,
  reset: resetTask,
} = useAsyncTask<TaskStatus>(submitCurrentTask, fetchConvertTaskStatus, {
  onCompleted(s) {
    appStore.setCompletedTask(s)
    window.umami?.track('convert-complete', {
      inputType: inputType.value,
      has3mf: Boolean(s.result?.has_3mf),
      elapsed_s: s.elapsed_ms != null ? Math.round(s.elapsed_ms / 1000) : -1,
      width: s.result?.input_width ?? 0,
      height: s.result?.input_height ?? 0,
      avg_de: s.result?.stats?.avg_db_de ?? -1,
    })
  },
  onFailed(s) {
    appStore.markTaskFailed()
    window.umami?.track('convert-fail', {
      inputType: inputType.value,
      error: (s.error ?? 'unknown').slice(0, 120),
    })
  },
})

const displayError = computed(() => (error.value ? mapTaskError(error.value, t) : null))

watch(selectedFile, () => {
  resetTask()
})

const isRunning = computed(() => {
  const s = taskStatus.value?.status
  return s === 'pending' || s === 'running'
})

const progressPercent = computed(() => {
  if (!taskStatus.value) return 0
  const overall = computeOverallProgress(taskStatus.value.stage, taskStatus.value.progress)
  return Math.round(overall * 100)
})

const stageText = computed(() => {
  if (!taskStatus.value) return ''
  return getStageLabel(taskStatus.value.stage)
})

const taskState = computed(() => taskStatus.value?.status ?? '')
const showRunningProgress = computed(() => taskState.value === 'running')

const convertButtonText = computed(() => {
  if (taskState.value === 'pending') return t('convert.queuing')
  if (showRunningProgress.value)
    return `${stageText.value || t('convert.stages.unknown')} ${progressPercent.value}%`
  if (loading.value) return t('convert.submitTask')
  return t('convert.startConvert')
})

const canSubmit = computed(() => {
  return selectedFile.value !== null && !loading.value
})

const taskWarnings = computed(() => completedTask.value?.warnings ?? [])
const result = computed(() => completedTask.value?.result ?? null)
const completedTaskId = computed(() => completedTask.value?.id ?? '')
const isDownloading3mf = ref(false)
const canDownload3mf = computed(() => {
  return (
    completedTask.value?.status === 'completed' &&
    Boolean(result.value?.has_3mf) &&
    Boolean(completedTaskId.value)
  )
})

const isVectorInput = computed(() => inputType.value === 'vector')

const vectorFileSizeText = computed(() => {
  if (!isVectorInput.value || !selectedFile.value) return ''
  const kb = selectedFile.value.size / 1024
  if (kb < 1024) return `${kb.toFixed(0)} KB`
  return `${(kb / 1024).toFixed(1)} MB`
})

const download3mfFilename = computed(() => {
  const rawName = selectedFile.value?.name?.trim() ?? ''
  if (rawName.length > 0) {
    const dot = rawName.lastIndexOf('.')
    const baseName = dot > 0 ? rawName.slice(0, dot) : rawName
    return `${baseName || 'result'}.3mf`
  }
  return completedTaskId.value ? `${completedTaskId.value.substring(0, 8)}.3mf` : 'result.3mf'
})

async function handleConvert() {
  if (!selectedFile.value) return
  window.umami?.track('convert-start', {
    inputType: inputType.value,
    color_layers: params.value.color_layers ?? 0,
    model_enable: Boolean(params.value.model_enable),
  })
  appStore.markTaskStarted()
  await submitTask()
}

function submitMatchOnly() {
  const file = selectedFile.value!
  return submitMatchOnlyTask(file, params.value, inputType.value)
}

const {
  status: matchStatus,
  loading: matchLoading,
  error: matchError,
  submit: submitMatch,
  reset: resetMatch,
} = useAsyncTask<TaskStatus>(submitMatchOnly, fetchConvertTaskStatus, {
  onCompleted(s) {
    appStore.setCompletedTask(s)
    appStore.setRecipeEditorTaskId(s.id)
    window.umami?.track('match-preview-complete', {
      inputType: inputType.value,
      elapsed_s: s.elapsed_ms != null ? Math.round(s.elapsed_ms / 1000) : -1,
      width: s.result?.input_width ?? 0,
      height: s.result?.input_height ?? 0,
      avg_de: s.result?.stats?.avg_db_de ?? -1,
    })
    window.umami?.track('recipe-editor-open')
  },
  onFailed(s) {
    appStore.markTaskFailed()
    window.umami?.track('match-preview-fail', {
      inputType: inputType.value,
      error: (s.error ?? 'unknown').slice(0, 120),
    })
  },
})

const displayMatchError = computed(() =>
  matchError.value ? mapTaskError(matchError.value, t) : null,
)

watch(selectedFile, () => {
  resetMatch()
})

const matchIsRunning = computed(() => {
  const s = matchStatus.value?.status
  return s === 'pending' || s === 'running'
})

const matchProgressPercent = computed(() => {
  if (!matchStatus.value) return 0
  const overall = computeOverallProgress(matchStatus.value.stage, matchStatus.value.progress)
  return Math.round(overall * 100)
})

const canMatchPreview = computed(() => {
  return selectedFile.value !== null && !loading.value && !matchLoading.value
})

async function handleMatchPreview() {
  if (!selectedFile.value) return
  window.umami?.track('match-preview-start', {
    inputType: inputType.value,
    color_layers: params.value.color_layers ?? 0,
    model_enable: Boolean(params.value.model_enable),
  })
  appStore.markTaskStarted()
  await submitMatch()
}

async function handleDownload3MF() {
  if (!canDownload3mf.value || isDownloading3mf.value) return
  isDownloading3mf.value = true
  const filename = download3mfFilename.value
  try {
    await downloadByUrl(getResultPath(completedTaskId.value), filename)
    window.umami?.track('download-3mf', { filename })
  } catch {
    // error is already handled by runtime abstraction caller when needed
  } finally {
    isDownloading3mf.value = false
  }
}
</script>

<template>
  <NCard size="small">
    <NSpace vertical :size="12">
      <NSpace vertical :size="8">
        <NButton
          block
          size="large"
          type="info"
          ghost
          :disabled="!canMatchPreview"
          :loading="matchLoading"
          @click="handleMatchPreview"
        >
          {{
            matchIsRunning
              ? `${t('convert.matchPreview')} ${matchProgressPercent}%`
              : t('convert.matchPreview')
          }}
        </NButton>
        <ProgressActionGroup
          :primary-label="convertButtonText"
          :primary-disabled="!canSubmit"
          :primary-show-progress="showRunningProgress"
          :primary-progress-percent="progressPercent"
          :secondary-visible="completedTask?.status === 'completed' && Boolean(result?.has_3mf)"
          :secondary-label="t('convert.download3mf')"
          secondary-type="success"
          :secondary-disabled="!canDownload3mf || isDownloading3mf"
          :secondary-loading="isDownloading3mf"
          @primary-click="handleConvert"
          @secondary-click="handleDownload3MF"
        />
        <NText v-if="!selectedFile" depth="3" style="font-size: 13px">
          {{ t('convert.uploadFirst') }}
        </NText>
      </NSpace>

      <NAlert v-if="isVectorInput && !isRunning && !isCompleted" type="warning">
        {{
          t('convert.svgInputHint', {
            sizeInfo: vectorFileSizeText ? `, ${vectorFileSizeText}` : '',
          })
        }}
      </NAlert>

      <NAlert v-if="displayError" type="error" closable @close="error = null">
        {{ displayError }}
      </NAlert>
      <NAlert v-if="displayMatchError" type="error" closable @close="matchError = null">
        {{ displayMatchError }}
      </NAlert>
      <NAlert v-for="w in taskWarnings" :key="w" type="warning" closable>
        {{ t(`convert.warnings.${w}`, t(`convert.warnings.${w}`)) }}
      </NAlert>
    </NSpace>
  </NCard>
</template>
