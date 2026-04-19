import i18n from '../locales'
import { getRuntimeKind, isElectronRuntime } from '../runtime/platform'

// ---------------------------------------------------------------------------
// Event catalog
//
// All custom events MUST be registered here so that:
//   - `EventName` is a closed union, preventing typos / drift at call sites.
//   - `EventPayloadMap` provides compile-time shape checking for each event.
// ---------------------------------------------------------------------------

export const EVENT_NAMES = [
  // Convert (raster / vector → 3MF)
  'image-select',
  'convert-start',
  'convert-complete',
  'convert-fail',
  'match-preview-start',
  'match-preview-complete',
  'match-preview-fail',
  'convert-download-3mf',
  // Recipe editor / model regeneration
  'recipe-open',
  'recipe-replace',
  'recipe-undo',
  'recipe-global-mode-toggle',
  'recipe-generate-complete',
  'recipe-generate-fail',
  'recipe-download-3mf',
  // Vectorize
  'vectorize-start',
  'vectorize-complete',
  'vectorize-fail',
  'vectorize-download-svg',
  'vectorize-use-for-convert',
  'vectorize-clear-file',
  // Matting (background removal)
  'matting-model-download-start',
  'matting-model-download-complete',
  'matting-model-download-fail',
  'matting-model-download-cancel',
  'matting-connectivity-check',
  'matting-start',
  'matting-complete',
  'matting-fail',
  'matting-postprocess-done',
  'matting-download-mask',
  'matting-download-foreground',
  'matting-use-for-convert',
  // Calibration (board generation)
  'calibration-generate-start',
  'calibration-generate-complete',
  'calibration-generate-fail',
  'calibration-download-3mf',
  'calibration-download-meta',
  // ColorDB build / upload / delete
  'colordb-locate-start',
  'colordb-locate-complete',
  'colordb-locate-fail',
  'colordb-build-start',
  'colordb-build-complete',
  'colordb-build-fail',
  'colordb-download-json',
  'colordb-upload',
  'colordb-delete',
  // UI interactions
  'locale-toggle',
  'theme-toggle',
  'tutorial-click',
  'github-click',
  'makerworld-click',
  'whats-new-open',
  'announcement-dismiss',
  'update-banner-click',
  'check-update-click',
  // Lifecycle telemetry
  'memory-status',
] as const

export type EventName = (typeof EVENT_NAMES)[number]

type InputKind = 'raster' | 'vector'

type BulkMode = 'single' | 'batch'

type EmptyPayload = Record<string, never>

export interface EventPayloadMap {
  // ---- Convert ----
  'image-select': { input_type: InputKind }
  'convert-start': {
    input_type: InputKind
    color_layers: number
    model_enable: boolean
    input_size_kb: number
    colordb_count: number
  }
  'convert-complete': {
    input_type: InputKind
    has_3mf: boolean
    duration_ms: number
    width: number
    height: number
    avg_de: number
    colordb_count: number
  }
  'convert-fail': {
    input_type: InputKind
    error: string
    error_code: string
  }
  'match-preview-start': {
    input_type: InputKind
    color_layers: number
    model_enable: boolean
    input_size_kb: number
    colordb_count: number
  }
  'match-preview-complete': {
    input_type: InputKind
    duration_ms: number
    width: number
    height: number
    avg_de: number
  }
  'match-preview-fail': {
    input_type: InputKind
    error: string
    error_code: string
  }
  'convert-download-3mf': { filename: string }

  // ---- Recipe ----
  'recipe-open': EmptyPayload
  'recipe-replace': { region_count: number }
  'recipe-undo': EmptyPayload
  'recipe-global-mode-toggle': { enabled: boolean }
  'recipe-generate-complete': { duration_ms: number }
  'recipe-generate-fail': { error: string; error_code: string }
  'recipe-download-3mf': { filename: string }

  // ---- Vectorize ----
  'vectorize-start': { input_size_kb: number; num_colors: number }
  'vectorize-complete': {
    duration_ms: number
    width: number
    height: number
    num_shapes: number
    svg_size_kb: number
    resolved_num_colors: number
  }
  'vectorize-fail': { error: string; error_code: string }
  'vectorize-download-svg': { svg_size_kb: number }
  'vectorize-use-for-convert': EmptyPayload
  'vectorize-clear-file': EmptyPayload

  // ---- Matting ----
  'matting-model-download-start': { pending_count: number; total_count: number }
  'matting-model-download-complete': { duration_ms: number; total_count: number }
  'matting-model-download-fail': { error: string; error_code: string }
  'matting-model-download-cancel': EmptyPayload
  'matting-connectivity-check': {
    available_sources: number
    total_sources: number
    duration_ms: number
  }
  'matting-start': { method: string; input_size_kb: number }
  'matting-complete': {
    method: string
    duration_ms: number
    has_alpha: boolean
  }
  'matting-fail': { method: string; error: string; error_code: string }
  'matting-postprocess-done': { duration_ms: number }
  'matting-download-mask': EmptyPayload
  'matting-download-foreground': EmptyPayload
  'matting-use-for-convert': EmptyPayload

  // ---- Calibration ----
  'calibration-generate-start': {
    board_index: number
    channel_count: number
    nozzle_size: string
    face_orientation: string
  }
  'calibration-generate-complete': {
    board_index: number
    channel_count: number
    nozzle_size: string
    face_orientation: string
    duration_ms: number
  }
  'calibration-generate-fail': {
    board_index: number
    channel_count: number
    error: string
    error_code: string
  }
  'calibration-download-3mf': {
    board_index: number
    channel_count: number
  }
  'calibration-download-meta': {
    board_index: number
    channel_count: number
  }

  // ---- ColorDB ----
  'colordb-locate-start': EmptyPayload
  'colordb-locate-complete': { duration_ms: number }
  'colordb-locate-fail': { error: string; error_code: string }
  'colordb-build-start': EmptyPayload
  'colordb-build-complete': {
    num_channels: number
    num_entries: number
    duration_ms: number
  }
  'colordb-build-fail': { error: string; error_code: string }
  'colordb-download-json': EmptyPayload
  'colordb-upload': {
    mode: BulkMode
    count: number
    ok: number
    fail: number
  }
  'colordb-delete': {
    mode: BulkMode
    ok: number
    fail: number
  }

  // ---- UI ----
  'locale-toggle': { from: string; to: string }
  'theme-toggle': { theme: 'dark' | 'light' }
  'tutorial-click': EmptyPayload
  'github-click': { target: 'repo' | 'issues' }
  'makerworld-click': EmptyPayload
  'whats-new-open': { trigger: 'auto' | 'manual' }
  'announcement-dismiss': { id: string }
  'update-banner-click': { action: 'download' | 'dismiss' }
  'check-update-click': { result: 'has-update' | 'no-update' | 'fail' }

  // ---- Lifecycle ----
  'memory-status': {
    rss_mb: number
    heap_mb: number
    artifact_pct: number
    usage_pct: number
    allocator: string
  }
}

// ---------------------------------------------------------------------------
// Internal Umami binding (intentionally not declared on `Window`)
// ---------------------------------------------------------------------------

type UmamiTracker = {
  track: {
    (): void
    (event: string, data?: Record<string, unknown>): void
    (callback: (props: Record<string, unknown>) => Record<string, unknown>): void
  }
  identify: {
    (data: Record<string, unknown>): void
    (id: string, data?: Record<string, unknown>): void
  }
}

function getUmami(): UmamiTracker | undefined {
  if (typeof window === 'undefined') return undefined
  return (window as unknown as { umami?: UmamiTracker }).umami
}

// ---------------------------------------------------------------------------
// Session identity
//
// A stable, fully anonymous per-install UUID stored in localStorage. No PII.
// On Electron we do not load the tracker at all, so this key is unused there.
// ---------------------------------------------------------------------------

const DISTINCT_ID_KEY = 'chromaprint3d-analytics-id'

export interface SessionProperties {
  runtime: 'electron' | 'browser'
  channel: 'stable' | 'preview' | 'unknown'
  version: string
  locale: string
}

function readCurrentLocale(): string {
  const raw = i18n.global.locale as unknown
  if (raw && typeof raw === 'object' && 'value' in (raw as { value?: unknown })) {
    const v = (raw as { value?: unknown }).value
    return typeof v === 'string' ? v : String(v ?? 'unknown')
  }
  return typeof raw === 'string' ? raw : 'unknown'
}

function detectChannel(version: string | undefined | null): SessionProperties['channel'] {
  if (!version) return 'unknown'
  if (/-rc\.\d+/i.test(version)) return 'preview'
  if (/^\d+\.\d+\.\d+$/.test(version)) return 'stable'
  return 'unknown'
}

const bootVersion =
  typeof __APP_VERSION__ === 'string' && __APP_VERSION__ ? __APP_VERSION__ : 'unknown'

let currentSessionProps: SessionProperties = {
  runtime: getRuntimeKind(),
  channel: detectChannel(bootVersion),
  version: bootVersion,
  locale: readCurrentLocale(),
}

function getOrCreateDistinctId(): string | null {
  if (typeof localStorage === 'undefined') return null
  try {
    const existing = localStorage.getItem(DISTINCT_ID_KEY)
    if (existing) return existing
    const id =
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`
    localStorage.setItem(DISTINCT_ID_KEY, id)
    return id
  } catch {
    return null
  }
}

// The Umami snippet is loaded via `<script defer>` so it typically becomes
// available before Vue's `onMounted` hooks fire, but there is still a window
// in which early calls could race the script. To avoid losing the very first
// pageview (triggered on mount) or the `identify` payload, we buffer calls
// and flush them once `window.umami` is detected. The buffer has a bounded
// retry window so failures are forgotten rather than leaking memory.

const PENDING_MAX_QUEUE = 64
const PENDING_MAX_ATTEMPTS = 40 // ~10s total at 250ms
const pendingCalls: Array<(u: UmamiTracker) => void> = []
let pendingTimer: ReturnType<typeof setTimeout> | null = null
let pendingAttempts = 0
let identifyNeeded = false

function flushPending(): boolean {
  const umami = getUmami()
  if (!umami) return false
  if (identifyNeeded && typeof umami.identify === 'function') {
    identifyNeeded = false
    const id = getOrCreateDistinctId()
    try {
      if (id) {
        umami.identify(id, { ...currentSessionProps })
      } else {
        umami.identify({ ...currentSessionProps })
      }
    } catch {
      // swallow: analytics must never break the app
    }
  }
  while (pendingCalls.length > 0) {
    const call = pendingCalls.shift()
    if (!call) break
    try {
      call(umami)
    } catch {
      // ignore
    }
  }
  return true
}

function schedulePendingFlush() {
  if (flushPending()) return
  if (pendingTimer != null) return
  pendingAttempts = 0
  const retry = () => {
    pendingTimer = null
    pendingAttempts += 1
    if (flushPending()) return
    if (pendingAttempts >= PENDING_MAX_ATTEMPTS) {
      // Give up without re-queueing; further calls will restart the loop.
      pendingCalls.length = 0
      identifyNeeded = false
      return
    }
    pendingTimer = setTimeout(retry, 250)
  }
  pendingTimer = setTimeout(retry, 250)
}

function enqueuePending(fn: (u: UmamiTracker) => void) {
  if (pendingCalls.length >= PENDING_MAX_QUEUE) return
  pendingCalls.push(fn)
  schedulePendingFlush()
}

function scheduleIdentify() {
  identifyNeeded = true
  schedulePendingFlush()
}

export function identifyAnonymousUser() {
  scheduleIdentify()
}

export function updateSessionProperties(partial: Partial<SessionProperties>) {
  currentSessionProps = { ...currentSessionProps, ...partial }
  if (partial.version && (!partial.channel || partial.channel === 'unknown')) {
    currentSessionProps.channel = detectChannel(partial.version)
  }
  scheduleIdentify()
}

// ---------------------------------------------------------------------------
// Script injection
// ---------------------------------------------------------------------------

let initialized = false

export function initAnalytics() {
  if (initialized) return
  initialized = true
  if (typeof document === 'undefined') return
  // Desktop app is a private single-user environment; tracking browser-only
  // keeps the Electron binary free of third-party network calls.
  if (isElectronRuntime()) return

  const host = import.meta.env.VITE_UMAMI_HOST
  const websiteId = import.meta.env.VITE_UMAMI_WEBSITE_ID
  if (!host || !websiteId) return

  const script = document.createElement('script')
  script.defer = true
  script.src = `${host}/script.js`
  script.dataset.websiteId = websiteId
  script.dataset.hostUrl = host
  script.dataset.domains = window.location.hostname
  script.dataset.performance = 'true'
  script.dataset.doNotTrack = 'true'
  script.dataset.excludeSearch = 'true'
  script.dataset.excludeHash = 'true'
  script.addEventListener('load', () => scheduleIdentify(), { once: true })
  document.head.appendChild(script)
}

// ---------------------------------------------------------------------------
// Normalization utilities
// ---------------------------------------------------------------------------

/** Round byte count to an integer kilobyte value. Returns 0 for invalid input. */
export function roundKb(bytes: number | null | undefined): number {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes <= 0) return 0
  return Math.round(bytes / 1024)
}

/** Normalize a millisecond elapsed time to a non-negative integer. Returns -1 if unknown. */
export function toDurationMs(ms: number | null | undefined): number {
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 0) return -1
  return Math.round(ms)
}

/** Umami caps strings at 500 chars; we keep messages short so payloads stay readable. */
export function shortenError(e: unknown): string {
  const raw = e instanceof Error ? e.message : typeof e === 'string' ? e : String(e ?? '')
  const trimmed = raw.trim()
  if (!trimmed) return 'unknown'
  return trimmed.length > 120 ? trimmed.slice(0, 120) : trimmed
}

/**
 * Reduce an arbitrary error into a low-cardinality bucket suitable for
 * aggregation in Umami. Prefer narrow error codes for dashboarding over the
 * raw `error` string.
 */
export function resolveErrorCode(e: unknown): string {
  const raw = e instanceof Error ? e.message : typeof e === 'string' ? e : String(e ?? '')
  if (!raw) return 'unknown'
  const status = /\b(?:http[\s:]*|status[\s:=]*)(\d{3})\b/i.exec(raw)
  if (status) return `http_${status[1]}`
  const bareStatus = /\b([45]\d{2})\b/.exec(raw)
  if (bareStatus) return `http_${bareStatus[1]}`
  const lowered = raw.toLowerCase()
  if (lowered.includes('aborted') || lowered.includes('cancel')) return 'cancelled'
  if (lowered.includes('timeout')) return 'timeout'
  if (lowered.includes('network') || lowered.includes('failed to fetch')) return 'network'
  if (lowered.includes('not found')) return 'not_found'
  if (lowered.includes('unauthorized')) return 'unauthorized'
  if (lowered.includes('forbidden')) return 'forbidden'
  if (lowered.includes('parse') || lowered.includes('json') || lowered.includes('syntax')) {
    return 'parse_error'
  }
  if (lowered.includes('payload') || lowered.includes('too large')) return 'payload_too_large'
  if (lowered.includes('unsupported') || lowered.includes('invalid')) return 'invalid_input'
  return 'unknown'
}

// ---------------------------------------------------------------------------
// Public tracking API
// ---------------------------------------------------------------------------

type EmptyEventName = {
  [K in EventName]: EventPayloadMap[K] extends EmptyPayload ? K : never
}[EventName]

type PayloadEventName = Exclude<EventName, EmptyEventName>

export function trackEvent<K extends EmptyEventName>(name: K): void
export function trackEvent<K extends PayloadEventName>(name: K, data: EventPayloadMap[K]): void
export function trackEvent<K extends EventName>(name: K, data?: EventPayloadMap[K]): void {
  if (isElectronRuntime()) return
  const payload = data as Record<string, unknown> | undefined
  const send = (u: UmamiTracker) => {
    if (payload && Object.keys(payload).length > 0) {
      u.track(name, payload)
    } else {
      u.track(name)
    }
  }
  const umami = getUmami()
  if (umami) {
    try {
      send(umami)
    } catch {
      // never let analytics break the app
    }
    return
  }
  enqueuePending(send)
}

export function trackPageview(url: string, title?: string) {
  if (isElectronRuntime()) return
  const send = (u: UmamiTracker) => {
    u.track((props) => ({
      ...props,
      url,
      ...(title ? { title } : {}),
    }))
  }
  const umami = getUmami()
  if (umami) {
    try {
      send(umami)
    } catch {
      // noop
    }
    return
  }
  enqueuePending(send)
}
