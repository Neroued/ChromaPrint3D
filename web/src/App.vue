<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import {
  NConfigProvider,
  NLayout,
  NLayoutHeader,
  NLayoutContent,
  NLayoutFooter,
  NSpace,
  NGrid,
  NGridItem,
  NText,
  NTag,
  NTabs,
  NTabPane,
  NMessageProvider,
} from 'naive-ui'
import ImageUpload from './components/ImageUpload.vue'
import ParamPanel from './components/ParamPanel.vue'
import ConvertPanel from './components/ConvertPanel.vue'
import ResultPanel from './components/ResultPanel.vue'
import CalibrationPanel from './components/CalibrationPanel.vue'
import { fetchHealth } from './api'
import type { ConvertParams, TaskStatus } from './types'

const selectedFile = ref<File | null>(null)
const params = ref<ConvertParams>({})
const completedTask = ref<TaskStatus | null>(null)
const activeTab = ref('convert')
const colordbVersion = ref(0)

const serverOnline = ref(false)
const serverVersion = ref('')
const activeTasks = ref(0)
const totalTasks = ref(0)
let healthTimer: ReturnType<typeof setInterval> | null = null

async function checkHealth() {
  try {
    const h = await fetchHealth()
    serverOnline.value = h.status === 'ok'
    serverVersion.value = h.version ?? ''
    activeTasks.value = h.active_tasks ?? 0
    totalTasks.value = h.total_tasks ?? 0
  } catch {
    serverOnline.value = false
  }
}

function handleTaskCompleted(task: TaskStatus) {
  completedTask.value = task
}

function handleTaskFailed(_task: TaskStatus) {
  completedTask.value = null
}

function handleColorDBUpdated() {
  colordbVersion.value++
}

onMounted(() => {
  checkHealth()
  healthTimer = setInterval(checkHealth, 15000)
})

onUnmounted(() => {
  if (healthTimer) {
    clearInterval(healthTimer)
  }
})
</script>

<template>
  <NConfigProvider>
    <NMessageProvider>
      <NLayout style="min-height: 100vh; background: #f5f5f5">
        <!-- Header -->
        <NLayoutHeader
          bordered
          style="
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #fff;
          "
        >
          <NSpace align="center" :size="12">
            <NText strong style="font-size: 20px; letter-spacing: 0.5px">
              ChromaPrint3D
            </NText>
            <NText v-if="serverVersion" depth="3" style="font-size: 12px">
              v{{ serverVersion }}
            </NText>
          </NSpace>
          <NSpace align="center" :size="12">
            <NText v-if="serverOnline && totalTasks > 0" depth="3" style="font-size: 12px">
              {{ activeTasks > 0 ? `${activeTasks} 个任务进行中` : `${totalTasks} 个历史任务` }}
            </NText>
            <NTag
              :type="serverOnline ? 'success' : 'error'"
              size="small"
              round
            >
              {{ serverOnline ? '服务器在线' : '服务器离线' }}
            </NTag>
          </NSpace>
        </NLayoutHeader>

        <!-- Main content -->
        <NLayoutContent style="padding: 24px; max-width: 1200px; margin: 0 auto">
          <NTabs v-model:value="activeTab" type="line" size="large" animated>
            <NTabPane name="convert" tab="图像转换" display-directive="show">
              <NSpace vertical :size="16" style="padding-top: 16px">
                <NGrid :cols="2" :x-gap="16" responsive="screen" item-responsive>
                  <NGridItem span="2 m:1">
                    <ImageUpload v-model="selectedFile" />
                  </NGridItem>
                  <NGridItem span="2 m:1">
                    <ParamPanel v-model="params" :refresh-trigger="colordbVersion" />
                  </NGridItem>
                </NGrid>
                <ConvertPanel
                  :file="selectedFile"
                  :params="params"
                  @task-completed="handleTaskCompleted"
                  @task-failed="handleTaskFailed"
                />
                <ResultPanel :task="completedTask" />
              </NSpace>
            </NTabPane>

            <NTabPane name="calibration" tab="校准工具（四色以下）" display-directive="show">
              <div style="padding-top: 16px">
                <CalibrationPanel @colordb-updated="handleColorDBUpdated" />
              </div>
            </NTabPane>
          </NTabs>
        </NLayoutContent>

        <!-- Footer -->
        <NLayoutFooter
          bordered
          style="
            padding: 16px 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #fff;
          "
        >
          <NSpace align="center" :size="16" justify="center" style="flex-wrap: wrap">
            <NText depth="3" style="font-size: 12px">
              ChromaPrint3D{{ serverVersion ? ` v${serverVersion}` : '' }}
            </NText>
            <NText depth="3" style="font-size: 12px">
              Multi-color 3D Print Image Processor
            </NText>
            <a
              href="https://github.com/neroued/ChromaPrint3D"
              target="_blank"
              rel="noopener noreferrer"
              class="footer-link"
            >
              <NText depth="3" style="font-size: 12px">GitHub</NText>
            </a>
            <a
              href="mailto:neroued@gmail.com"
              class="footer-link"
            >
              <NText depth="3" style="font-size: 12px">Neroued@gmail.com</NText>
            </a>
          </NSpace>
        </NLayoutFooter>
      </NLayout>
    </NMessageProvider>
  </NConfigProvider>
</template>

<style>
/* Global reset */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family:
    -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
    'Helvetica Neue', Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.footer-link {
  text-decoration: none;
  transition: opacity 0.2s;
}

.footer-link:hover {
  opacity: 0.7;
}
</style>
