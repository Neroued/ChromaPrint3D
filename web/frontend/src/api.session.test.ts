import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchHealth } from './api/health'
import {
  applySessionHeader,
  getSessionHeaderName,
  mergeSessionHeader,
  setSessionToken,
} from './runtime/session'

function buildOkResponse(data: unknown, headers?: Record<string, string>): Response {
  return new Response(JSON.stringify({ ok: true, data }), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      ...(headers ?? {}),
    },
  })
}

function readSessionHeader(): string | null {
  return mergeSessionHeader().get(getSessionHeaderName())
}

describe('api session token lifecycle', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    setSessionToken(null)
  })

  it('在响应不含会话头时保留已有 token', async () => {
    setSessionToken('session-A')
    const fetchMock = vi.fn().mockResolvedValue(buildOkResponse({ status: 'ok', version: 'test' }))
    vi.stubGlobal('fetch', fetchMock)

    await fetchHealth()

    const [, requestInit] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(mergeSessionHeader(requestInit.headers).get(getSessionHeaderName())).toBe('session-A')
    expect(readSessionHeader()).toBe('session-A')
  })

  it('在响应携带会话头时更新 token', async () => {
    setSessionToken('session-A')
    const headerName = getSessionHeaderName()
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        buildOkResponse({ status: 'ok', version: 'test' }, { [headerName]: 'session-B' }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await fetchHealth()

    const headers = new Headers()
    applySessionHeader(headers)
    expect(headers.get(headerName)).toBe('session-B')
  })
})
