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

export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetchWithSession(url, init)

  let envelope: ApiEnvelope<T> | null = null
  try {
    envelope = (await res.json()) as ApiEnvelope<T>
  } catch {
    // ignore parse errors, fallback to HTTP status
  }

  if (!res.ok) {
    throw new Error(parseErrorMessage(envelope?.error, `HTTP ${res.status}`))
  }
  if (!envelope?.ok) {
    throw new Error(parseErrorMessage(envelope?.error, 'Request failed'))
  }
  return envelope.data as T
}
