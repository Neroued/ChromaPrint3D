import { defineStore } from 'pinia'
import { ref } from 'vue'
import { fetchHealth } from '../api/health'
import { getStorageItem, setStorageItem } from '../runtime/storage'
import { detectSystemDarkMode } from '../runtime/theme'
import type { WritableComputedRef } from 'vue'
import i18n from '../locales'
import { LOCALE_STORAGE_KEY, type SupportedLocale } from '../locales'
import type { ConvertAnyParams, ImageDimensions, InputType, TaskStatus } from '../types'

export const THEME_STORAGE_KEY = 'chromaprint3d-theme'

export const useAppStore = defineStore('app', () => {
  const selectedFile = ref<File | null>(null)
  const imageDimensions = ref<ImageDimensions | null>(null)
  const inputType = ref<InputType>('raster')
  const params = ref<ConvertAnyParams>({})
  const completedTask = ref<TaskStatus | null>(null)
  const activeTab = ref('convert')
  const colordbVersion = ref(0)

  const serverOnline = ref(false)
  const serverVersion = ref('')
  const activeTasks = ref(0)
  const totalTasks = ref(0)

  const isDark = ref(false)

  function setSelectedFile(file: File | null) {
    selectedFile.value = file
    completedTask.value = null
  }

  function setImageDimensions(dims: ImageDimensions | null) {
    imageDimensions.value = dims
  }

  function setInputType(type: InputType) {
    inputType.value = type
  }

  function setParams(next: ConvertAnyParams) {
    params.value = next
  }

  function clearCompletedTask() {
    completedTask.value = null
  }

  function setCompletedTask(task: TaskStatus | null) {
    completedTask.value = task
  }

  function markTaskStarted() {
    completedTask.value = null
  }

  function markTaskFailed() {
    completedTask.value = null
  }

  function refreshColorDBs() {
    colordbVersion.value += 1
  }

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

  async function initTheme() {
    const stored = await getStorageItem(THEME_STORAGE_KEY)
    if (stored === 'dark') {
      isDark.value = true
      return
    }
    if (stored === 'light') {
      isDark.value = false
      return
    }
    isDark.value = await detectSystemDarkMode()
  }

  async function persistTheme(dark: boolean) {
    await setStorageItem(THEME_STORAGE_KEY, dark ? 'dark' : 'light')
  }

  const localeRef = i18n.global.locale as unknown as WritableComputedRef<SupportedLocale>

  async function initLocale() {
    const stored = await getStorageItem(LOCALE_STORAGE_KEY)
    if (stored === 'zh-CN' || stored === 'en') {
      localeRef.value = stored
    }
  }

  async function setLocale(loc: SupportedLocale) {
    localeRef.value = loc
    await setStorageItem(LOCALE_STORAGE_KEY, loc)
  }

  return {
    selectedFile,
    imageDimensions,
    inputType,
    params,
    completedTask,
    activeTab,
    colordbVersion,
    serverOnline,
    serverVersion,
    activeTasks,
    totalTasks,
    isDark,
    setSelectedFile,
    setImageDimensions,
    setInputType,
    setParams,
    clearCompletedTask,
    setCompletedTask,
    markTaskStarted,
    markTaskFailed,
    refreshColorDBs,
    checkHealth,
    initTheme,
    persistTheme,
    initLocale,
    setLocale,
  }
})
