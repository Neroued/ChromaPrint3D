import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchBlobWithSession, fetchTextWithSession, fetchWithSession } from './protectedRequest'
import { getSessionHeaderName, mergeSessionHeader, setSessionToken } from './session'

describe('runtime protectedRequest', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    setSessionToken(null)
  })

  it('fetchWithSession 应注入会话头并回写新 token', async () => {
    setSessionToken('session-A')
    const headerName = getSessionHeaderName()
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('{}', {
        status: 200,
        headers: { [headerName]: 'session-B' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await fetchWithSession('/api/v1/health')

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [requestUrl, requestInit] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(String(requestUrl)).toContain('/api/v1/health')
    expect(requestInit.credentials).toBe('include')
    expect(new Headers(requestInit.headers).get(headerName)).toBe('session-A')
    expect(mergeSessionHeader().get(headerName)).toBe('session-B')
  })

  it('fetchBlobWithSession 在成功时返回 blob', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('blob-body', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const blob = await fetchBlobWithSession('/api/v1/tasks/t1/artifacts/result')

    expect(blob.size).toBeGreaterThan(0)
    expect(await blob.text()).toBe('blob-body')
  })

  it('fetchTextWithSession 在非 2xx 时抛出 HTTP 错误', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('nope', { status: 404 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(fetchTextWithSession('/api/v1/tasks/t1/artifacts/svg')).rejects.toThrow('HTTP 404')
  })
})
