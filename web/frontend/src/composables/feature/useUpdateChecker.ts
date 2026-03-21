import { ref, computed, readonly } from 'vue'
import { useAppStore } from '../../stores/app'
import { isElectronRuntime } from '../../runtime/platform'
import { compareVersions, fetchManifestFromUrl } from './versionManifest'
export type { VersionManifest } from './versionManifest'

const DISMISS_KEY = 'chromaprint3d-update-dismissed'

function getManifestUrl(): string {
  const electronUrl = window.electron?.env?.updateManifestUrl
  if (electronUrl?.trim()) return electronUrl.trim()
  const envUrl = import.meta.env.VITE_UPDATE_MANIFEST_URL
  return typeof envUrl === 'string' ? envUrl.trim() : ''
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
const lastCheckFailed = ref(false)

let initialized = false

export function useUpdateChecker() {
  const appStore = useAppStore()

  const currentVersion = computed(() => {
    if (isElectronRuntime()) return __APP_VERSION__
    return appStore.serverVersion || __APP_VERSION__
  })

  const showBanner = computed(() => hasUpdate.value && !dismissed.value)

  async function fetchManifest() {
    return fetchManifestFromUrl(getManifestUrl())
  }

  async function checkForUpdate(): Promise<boolean> {
    if (checking.value) return false
    checking.value = true
    try {
      const manifest = await fetchManifest()
      if (!manifest) {
        lastCheckFailed.value = true
        return false
      }

      lastCheckFailed.value = false
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
    lastCheckFailed: readonly(lastCheckFailed),
    showBanner,
    currentVersion,
    checkForUpdate,
    dismiss,
    initOnce,
  }
}
