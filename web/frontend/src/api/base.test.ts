import { afterEach, describe, expect, it, vi } from 'vitest'
import { request } from './base'
import { setSessionToken } from '../runtime/session'

describe('request()', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    setSessionToken(null)
  })

  function mockFetch(envelope: object, status = 200) {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(envelope), {
        status,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)
    return fetchMock
  }

  it('returns data on success', async () => {
    mockFetch({ ok: true, data: { databases: [] } })
    const result = await request('/api/v1/databases')
    expect(result).toEqual({ databases: [] })
  })

  it('throws on HTTP error', async () => {
    mockFetch({ ok: false, error: 'bad request' }, 400)
    await expect(request('/api/v1/convert/raster', { method: 'POST' })).rejects.toThrow(
      'bad request',
    )
  })

  it('throws on envelope error', async () => {
    mockFetch({ ok: false, error: { message: 'not found' } })
    await expect(request('/api/v1/tasks/some-id')).rejects.toThrow('not found')
  })
})
