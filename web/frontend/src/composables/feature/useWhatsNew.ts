import { ref, computed, readonly } from 'vue'
import { useI18n } from 'vue-i18n'
import { fetchManifestFromUrl } from './versionManifest'
import type { VersionManifest } from './versionManifest'
import { getStorageItem, setStorageItem } from '../../runtime/storage'

const SEEN_KEY = 'chromaprint3d-whats-new-seen'
const LOCAL_MANIFEST_PATH = `${import.meta.env.BASE_URL}version-manifest.json`

const manifest = ref<VersionManifest | null>(null)
const visible = ref(false)
let initialized = false

function parseChangelogLines(raw: string): string[] {
  if (!raw) return []
  return raw
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => l.replace(/^[-·•]\s*/, ''))
}

export function useWhatsNew() {
  const { locale } = useI18n()

  const version = computed(() => manifest.value?.version ?? '')

  const changelogLines = computed(() => {
    if (!manifest.value) return []
    const raw =
      locale.value === 'en' && manifest.value.changelog_en
        ? manifest.value.changelog_en
        : manifest.value.changelog
    return parseChangelogLines(raw ?? '')
  })

  function show() {
    if (manifest.value && changelogLines.value.length > 0) {
      visible.value = true
    }
  }

  async function markAsSeen() {
    visible.value = false
    if (manifest.value?.version) {
      await setStorageItem(SEEN_KEY, manifest.value.version)
    }
  }

  async function initOnce() {
    if (initialized) return
    initialized = true

    const data = await fetchManifestFromUrl(LOCAL_MANIFEST_PATH)
    if (!data) return
    manifest.value = data

    if (!data.changelog?.trim()) return

    const seen = await getStorageItem(SEEN_KEY)
    if (seen !== data.version) {
      visible.value = true
    }
  }

  return {
    visible,
    version: readonly(version),
    changelogLines,
    show,
    markAsSeen,
    initOnce,
  }
}
