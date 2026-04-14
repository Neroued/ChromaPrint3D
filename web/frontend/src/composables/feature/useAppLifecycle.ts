import { onMounted, onUnmounted, watch } from 'vue'
import { useAppStore } from '../../stores/app'
import type { HealthMemory } from '../../types'
import { useLeaderLease } from './useLeaderLease'

let themeSwitchCleanupTimer: ReturnType<typeof setTimeout> | null = null

function applyDocumentTheme(dark: boolean) {
  if (typeof document === 'undefined') return
  const root = document.documentElement
  root.classList.add('theme-switching')
  root.setAttribute('data-theme', dark ? 'dark' : 'light')

  if (themeSwitchCleanupTimer) clearTimeout(themeSwitchCleanupTimer)
  themeSwitchCleanupTimer = setTimeout(() => {
    root.classList.remove('theme-switching')
    themeSwitchCleanupTimer = null
  }, 120)
}

async function syncTheme(dark: boolean, appStore: ReturnType<typeof useAppStore>) {
  applyDocumentTheme(dark)
  await appStore.persistTheme(dark)
  await window.electron?.theme?.setWindowBackground?.(dark)
}

const MEMORY_REPORT_INTERVAL_MS = 5 * 60 * 1000
const LEADER_CHANNEL_NAME = 'chromaprint3d-memory-reporter'
const MB = 1048576

let memoryReportTimer: ReturnType<typeof setInterval> | null = null

function reportMemoryToUmami(memory: HealthMemory, isLeader: boolean) {
  if (!isLeader || !window.umami) return
  window.umami.track('memory-status', {
    rss_mb: Math.round(memory.rss_bytes / MB),
    heap_mb: Math.round(memory.heap_resident_bytes / MB),
    artifact_pct:
      memory.artifact_budget_limit_bytes > 0
        ? Math.round((memory.artifact_budget_bytes / memory.artifact_budget_limit_bytes) * 100)
        : -1,
    usage_pct:
      memory.memory_limit_bytes > 0
        ? Math.round((memory.rss_bytes / memory.memory_limit_bytes) * 100)
        : -1,
    allocator: memory.allocator,
  })
}

function cleanupLeader() {
  if (memoryReportTimer) {
    clearInterval(memoryReportTimer)
    memoryReportTimer = null
  }
}

export function useAppLifecycle() {
  const appStore = useAppStore()
  let healthTimer: ReturnType<typeof setInterval> | null = null
  const { isLeader } = useLeaderLease({
    channelName: LEADER_CHANNEL_NAME,
  })

  onMounted(async () => {
    await appStore.initTheme()
    await appStore.initLocale()
    applyDocumentTheme(appStore.isDark)
    await window.electron?.theme?.setWindowBackground?.(appStore.isDark)
    await appStore.checkHealth()
    healthTimer = setInterval(() => {
      void appStore.checkHealth()
    }, 15000)

    memoryReportTimer = setInterval(() => {
      const m = appStore.serverMemory
      if (m) reportMemoryToUmami(m, isLeader.value)
    }, MEMORY_REPORT_INTERVAL_MS)
  })

  onUnmounted(() => {
    if (healthTimer) clearInterval(healthTimer)
    cleanupLeader()
    if (themeSwitchCleanupTimer) {
      clearTimeout(themeSwitchCleanupTimer)
      themeSwitchCleanupTimer = null
    }
    if (typeof document !== 'undefined') {
      document.documentElement.classList.remove('theme-switching')
    }
  })

  watch(
    () => appStore.isDark,
    (dark) => {
      void syncTheme(dark, appStore)
    },
  )
}
