import { fetchWithSession } from '../runtime/protectedRequest'

export type ApiErrorPayload =
  | {
      code?: string
      message?: string
    }
  | string

type ApiEnvelope<T> = {
  ok: boolean
  data?: T
  error?: ApiErrorPayload
}

function parseErrorMessage(payload: ApiErrorPayload | undefined, fallback: string): string {
  if (!payload) return fallback
  if (typeof payload === 'string') return payload
  return payload.message ?? payload.code ?? fallback
}

const UUID_RE = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi

export function normalizeEndpoint(url: string): string {
  return url
    .replace(/^\/api\/v1\//, '')
    .replace(UUID_RE, ':id')
    .replace(/\/artifacts\/[^/?]+/, '/artifacts/:key')
}

export function isAutomatedRequest(url: string, method: string): boolean {
  if (method === 'GET' && /^\/api\/v1\/tasks\/[^/]+$/.test(url)) return true
  if (url === '/api/v1/health') return true
  return false
}

export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const method = init?.method?.toUpperCase() ?? 'GET'
  const shouldTrack = !isAutomatedRequest(url, method)
  const start = shouldTrack ? performance.now() : 0

  let success = true
  try {
    const res = await fetchWithSession(url, init)

    let envelope: ApiEnvelope<T> | null = null
    try {
      envelope = (await res.json()) as ApiEnvelope<T>
    } catch {
      // ignore parse errors, fallback to HTTP status
    }

    if (!res.ok) {
      success = false
      throw new Error(parseErrorMessage(envelope?.error, `HTTP ${res.status}`))
    }
    if (!envelope?.ok) {
      success = false
      throw new Error(parseErrorMessage(envelope?.error, 'Request failed'))
    }
    return envelope.data as T
  } catch (err) {
    success = false
    throw err
  } finally {
    if (shouldTrack) {
      const duration = Math.round(performance.now() - start)
      const endpoint = normalizeEndpoint(url)
      window.umami?.track('api-call', { endpoint, method, success, duration })
    }
  }
}
