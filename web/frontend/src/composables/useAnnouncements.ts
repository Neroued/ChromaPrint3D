import { computed, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { fetchAnnouncements } from '../api/announcements'
import { useAppStore } from '../stores/app'
import { getStorageItem, setStorageItem } from '../runtime/storage'
import type { Announcement } from '../types'

const DISMISS_STORAGE_KEY = 'chromaprint3d-dismissed-announcements'

// Persisted as a JSON object: { [id]: updated_at }. We compare both the
// id AND the updated_at so that editing the announcement's content
// re-surfaces it for a user that had dismissed the previous revision.
type DismissMap = Record<string, string>

async function readDismissed(): Promise<DismissMap> {
  const raw = await getStorageItem(DISMISS_STORAGE_KEY)
  if (!raw) return {}
  try {
    const parsed = JSON.parse(raw) as unknown
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      const out: DismissMap = {}
      for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
        if (typeof v === 'string') out[k] = v
      }
      return out
    }
  } catch {
    // Corrupt or legacy storage — throw it away silently.
  }
  return {}
}

async function writeDismissed(value: DismissMap): Promise<void> {
  await setStorageItem(DISMISS_STORAGE_KEY, JSON.stringify(value))
}

let sharedState: ReturnType<typeof createAnnouncementsState> | null = null

function createAnnouncementsState() {
  const app = useAppStore()
  const { announcementsVersion } = storeToRefs(app)

  const announcements = ref<Announcement[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const dismissed = ref<DismissMap>({})
  const lastLoadedVersion = ref<string | null>(null)
  let hydrated = false

  async function hydrateDismissed() {
    if (hydrated) return
    dismissed.value = await readDismissed()
    hydrated = true
  }

  function isDismissed(a: Announcement): boolean {
    if (!a.dismissible) return false
    const prev = dismissed.value[a.id]
    if (!prev) return false
    return prev === a.updated_at
  }

  async function dismiss(id: string) {
    const match = announcements.value.find((a) => a.id === id)
    if (!match || !match.dismissible) return
    dismissed.value = { ...dismissed.value, [id]: match.updated_at }
    await writeDismissed(dismissed.value)
  }

  async function reload() {
    loading.value = true
    error.value = null
    try {
      const res = await fetchAnnouncements()
      announcements.value = res.announcements ?? []
      lastLoadedVersion.value = res.announcements_version ?? ''
      // Opportunistic compaction: drop dismiss entries for ids that no
      // longer exist on the server so storage does not grow forever.
      const active = new Set(announcements.value.map((a) => a.id))
      let changed = false
      const next: DismissMap = {}
      for (const [id, ts] of Object.entries(dismissed.value)) {
        if (active.has(id)) {
          next[id] = ts
        } else {
          changed = true
        }
      }
      if (changed) {
        dismissed.value = next
        await writeDismissed(next)
      }
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : 'Failed to load announcements'
    } finally {
      loading.value = false
    }
  }

  async function initialize() {
    await hydrateDismissed()
    await reload()
  }

  // Refetch when the health-reported version stamp actually changes.
  watch(announcementsVersion, async (next, prev) => {
    if (!next || next === prev) return
    if (next === lastLoadedVersion.value) return
    await reload()
  })

  const visible = computed(() => announcements.value.filter((a) => !isDismissed(a)))

  return {
    announcements,
    visible,
    loading,
    error,
    initialize,
    reload,
    dismiss,
    isDismissed,
  }
}

export function useAnnouncements() {
  if (!sharedState) sharedState = createAnnouncementsState()
  return sharedState
}

// Exposed for tests only — resets the module-level singleton.
export function __resetAnnouncementsStateForTests() {
  sharedState = null
}
