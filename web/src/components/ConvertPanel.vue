<script setup lang="ts">
import { ref, computed, onUnmounted } from 'vue'
import { NButton, NProgress, NCard, NText, NSpace, NAlert } from 'naive-ui'
import { submitConvert, fetchTaskStatus } from '../api'
import type { ConvertParams, TaskStatus } from '../types'

const props = defineProps<{
  file: File | null
  params: ConvertParams
}>()

const emit = defineEmits<{
  taskCompleted: [task: TaskStatus]
  taskFailed: [task: TaskStatus]
}>()

const taskId = ref<string | null>(null)
const taskStatus = ref<TaskStatus | null>(null)
const submitting = ref(false)
const error = ref<string | null>(null)
let pollTimer: ReturnType<typeof setInterval> | null = null

const stageLabels: Record<string, string> = {
  loading_resources: '加载资源...',
  processing_image: '处理图像...',
  matching: '颜色匹配...',
  building_model: '构建模型...',
  exporting: '导出结果...',
  unknown: '处理中...',
}

const isRunning = computed(() => {
  const s = taskStatus.value?.status
  return s === 'pending' || s === 'running'
})

const progressPercent = computed(() => {
  if (!taskStatus.value) return 0
  return Math.round(taskStatus.value.progress * 100)
})

const stageText = computed(() => {
  if (!taskStatus.value) return ''
  return stageLabels[taskStatus.value.stage] || taskStatus.value.stage
})

const canSubmit = computed(() => {
  return props.file !== null && !submitting.value && !isRunning.value
})

async function handleConvert() {
  if (!props.file) return
  error.value = null
  submitting.value = true
  taskStatus.value = null

  try {
    const resp = await submitConvert(props.file, props.params)
    taskId.value = resp.task_id
    startPolling()
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : '提交失败'
  } finally {
    submitting.value = false
  }
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(pollStatus, 500)
}

function stopPolling() {
  if (pollTimer !== null) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function pollStatus() {
  if (!taskId.value) return
  try {
    const status = await fetchTaskStatus(taskId.value)
    taskStatus.value = status

    if (status.status === 'completed') {
      stopPolling()
      emit('taskCompleted', status)
    } else if (status.status === 'failed') {
      stopPolling()
      error.value = status.error || '转换失败'
      emit('taskFailed', status)
    }
  } catch (e: unknown) {
    // Don't stop polling on transient network errors
    console.warn('Poll error:', e)
  }
}

onUnmounted(() => {
  stopPolling()
})
</script>

<template>
  <NCard size="small">
    <NSpace vertical :size="12">
      <NSpace align="center">
        <NButton
          type="primary"
          size="large"
          :loading="submitting"
          :disabled="!canSubmit"
          @click="handleConvert"
        >
          {{ isRunning ? '转换中...' : '开始转换' }}
        </NButton>

        <NText v-if="!file" depth="3" style="font-size: 13px">
          请先上传一张图片
        </NText>
      </NSpace>

      <NAlert v-if="error" type="error" closable @close="error = null">
        {{ error }}
      </NAlert>

      <div v-if="taskStatus && isRunning">
        <NSpace align="center" :size="8" style="margin-bottom: 4px">
          <NText style="font-size: 13px">{{ stageText }}</NText>
          <NText depth="3" style="font-size: 12px">{{ progressPercent }}%</NText>
        </NSpace>
        <NProgress
          type="line"
          :percentage="progressPercent"
          :show-indicator="false"
          :height="8"
          status="info"
          :processing="true"
        />
      </div>

      <div v-if="taskStatus?.status === 'completed'">
        <NProgress
          type="line"
          :percentage="100"
          :show-indicator="false"
          :height="8"
          status="success"
        />
        <NText type="success" style="font-size: 13px; margin-top: 4px; display: block">
          转换完成
        </NText>
      </div>
    </NSpace>
  </NCard>
</template>
