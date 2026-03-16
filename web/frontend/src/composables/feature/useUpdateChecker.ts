import { ref, computed, readonly } from 'vue'
import { useAppStore } from '../../stores/app'
import { isElectronRuntime } from '../../runtime/platform'

export type VersionManifest = {
  version: string
  download_url: string
  changelog: string
  published_at: string
}

const DISMISS_KEY = 'chromaprint3d-update-dismissed'
const FETCH_TIMEOUT_MS = 5000

function getManifestUrl(): string {
  const electronUrl = window.electron?.env?.updateManifestUrl
  if (electronUrl?.trim()) return electronUrl.trim()
  const envUrl = import.meta.env.VITE_UPDATE_MANIFEST_URL
  return typeof envUrl === 'string' ? envUrl.trim() : ''
}

function compareVersions(a: string, b: string): number {
  const pa = a.split('.').map(Number)
  const pb = b.split('.').map(Number)
  const len = Math.max(pa.length, pb.length)
  for (let i = 0; i < len; i++) {
    const na = pa[i] ?? 0
    const nb = pb[i] ?? 0
    if (na !== nb) return na - nb
  }
  return 0
}

function isValidManifest(data: unknown): data is VersionManifest {
  if (typeof data !== 'object' || data === null) return false
  const obj = data as Record<string, unknown>
  return typeof obj.version === 'string' && /^\d+\.\d+\.\d+/.test(obj.version)
}

function isDismissedForVersion(version: string): boolean {
  try {
    return sessionStorage.getItem(DISMISS_KEY) === version
  } catch {
    return false
  }
}

function setDismissedVersion(version: string) {
  try {
    sessionStorage.setItem(DISMISS_KEY, version)
  } catch {
    // ignore
  }
}

const hasUpdate = ref(false)
const latestVersion = ref('')
const changelog = ref('')
const downloadUrl = ref('')
const checking = ref(false)
const dismissed = ref(false)

let initialized = false

export function useUpdateChecker() {
  const appStore = useAppStore()

  const currentVersion = computed(() => {
    if (isElectronRuntime()) return __APP_VERSION__
    return appStore.serverVersion || __APP_VERSION__
  })

  const showBanner = computed(() => hasUpdate.value && !dismissed.value)

  async function fetchManifest(): Promise<VersionManifest | null> {
    const url = getManifestUrl()
    if (!url) return null

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)
    try {
      const resp = await fetch(url, { signal: controller.signal, cache: 'no-cache' })
      if (!resp.ok) return null
      const data: unknown = await resp.json()
      if (!isValidManifest(data)) return null
      return data
    } catch {
      return null
    } finally {
      clearTimeout(timer)
    }
  }

  async function checkForUpdate(): Promise<boolean> {
    if (checking.value) return false
    checking.value = true
    try {
      const manifest = await fetchManifest()
      if (!manifest) return false

      const current = currentVersion.value
      if (!current) return false

      if (compareVersions(manifest.version, current) > 0) {
        latestVersion.value = manifest.version
        changelog.value = manifest.changelog ?? ''
        downloadUrl.value = manifest.download_url ?? ''
        hasUpdate.value = true
        dismissed.value = isDismissedForVersion(manifest.version)
        return true
      }

      hasUpdate.value = false
      return false
    } catch {
      return false
    } finally {
      checking.value = false
    }
  }

  function dismiss() {
    dismissed.value = true
    if (latestVersion.value) {
      setDismissedVersion(latestVersion.value)
    }
  }

  function initOnce() {
    if (initialized) return
    initialized = true
    void checkForUpdate()
  }

  return {
    hasUpdate: readonly(hasUpdate),
    latestVersion: readonly(latestVersion),
    changelog: readonly(changelog),
    downloadUrl: readonly(downloadUrl),
    checking: readonly(checking),
    showBanner,
    currentVersion,
    checkForUpdate,
    dismiss,
    initOnce,
  }
}
