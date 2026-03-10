import { onMounted, onUnmounted, watch } from 'vue'
import { useAppStore } from '../../stores/app'

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

export function useAppLifecycle() {
  const appStore = useAppStore()
  let healthTimer: ReturnType<typeof setInterval> | null = null

  onMounted(async () => {
    await appStore.initTheme()
    await appStore.initLocale()
    applyDocumentTheme(appStore.isDark)
    await window.electron?.theme?.setWindowBackground?.(appStore.isDark)
    await appStore.checkHealth()
    healthTimer = setInterval(() => {
      void appStore.checkHealth()
    }, 15000)
  })

  onUnmounted(() => {
    if (healthTimer) clearInterval(healthTimer)
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
