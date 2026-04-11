import { afterEach, describe, expect, it, vi } from 'vitest'
import { isAutomatedRequest, normalizeEndpoint, request } from './base'
import { setSessionToken } from '../runtime/session'

describe('normalizeEndpoint', () => {
  it('strips the /api/v1/ prefix', () => {
    expect(normalizeEndpoint('/api/v1/databases')).toBe('databases')
  })

  it('replaces UUIDs with :id', () => {
    expect(normalizeEndpoint('/api/v1/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890')).toBe(
      'tasks/:id',
    )
  })

  it('replaces artifact keys with :key', () => {
    expect(
      normalizeEndpoint('/api/v1/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890/artifacts/layer-top'),
    ).toBe('tasks/:id/artifacts/:key')
  })

  it('handles recipe-editor paths', () => {
    expect(
      normalizeEndpoint('/api/v1/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890/recipe-editor/summary'),
    ).toBe('tasks/:id/recipe-editor/summary')
  })

  it('leaves paths without dynamic segments unchanged', () => {
    expect(normalizeEndpoint('/api/v1/convert/raster')).toBe('convert/raster')
    expect(normalizeEndpoint('/api/v1/convert/vector/analyze-width')).toBe(
      'convert/vector/analyze-width',
    )
  })

  it('handles multiple UUIDs in one path', () => {
    expect(
      normalizeEndpoint(
        '/api/v1/tasks/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/artifacts/ffffffff-1111-2222-3333-444444444444',
      ),
    ).toBe('tasks/:id/artifacts/:key')
  })
})

describe('isAutomatedRequest', () => {
  it('excludes task status polling (GET /api/v1/tasks/:id)', () => {
    expect(isAutomatedRequest('/api/v1/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'GET')).toBe(
      true,
    )
  })

  it('excludes health check', () => {
    expect(isAutomatedRequest('/api/v1/health', 'GET')).toBe(true)
  })

  it('allows task list (GET /api/v1/tasks)', () => {
    expect(isAutomatedRequest('/api/v1/tasks', 'GET')).toBe(false)
  })

  it('allows task deletion (DELETE /api/v1/tasks/:id)', () => {
    expect(isAutomatedRequest('/api/v1/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'DELETE')).toBe(
      false,
    )
  })

  it('allows POST conversion', () => {
    expect(isAutomatedRequest('/api/v1/convert/raster', 'POST')).toBe(false)
  })

  it('allows recipe-editor sub-paths', () => {
    expect(
      isAutomatedRequest(
        '/api/v1/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890/recipe-editor/summary',
        'GET',
      ),
    ).toBe(false)
  })
})

describe('request() tracking', () => {
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

  it('calls umami.track on successful non-automated request', async () => {
    mockFetch({ ok: true, data: { databases: [] } })
    const trackMock = vi.fn()
    vi.stubGlobal('umami', { track: trackMock })
    window.umami = { track: trackMock }

    await request('/api/v1/databases')

    expect(trackMock).toHaveBeenCalledTimes(1)
    const [event, data] = trackMock.mock.calls[0] as [string, Record<string, unknown>]
    expect(event).toBe('api-call')
    expect(data.endpoint).toBe('databases')
    expect(data.method).toBe('GET')
    expect(data.success).toBe(true)
    expect(typeof data.duration).toBe('number')
  })

  it('does not call umami.track for automated requests (polling)', async () => {
    mockFetch({ ok: true, data: { id: 't1', status: 'completed' } })
    const trackMock = vi.fn()
    window.umami = { track: trackMock }

    await request('/api/v1/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890')

    expect(trackMock).not.toHaveBeenCalled()
  })

  it('does not call umami.track for health check', async () => {
    mockFetch({ ok: true, data: { status: 'ok' } })
    const trackMock = vi.fn()
    window.umami = { track: trackMock }

    await request('/api/v1/health')

    expect(trackMock).not.toHaveBeenCalled()
  })

  it('reports success=false when request fails', async () => {
    mockFetch({ ok: false, error: 'bad request' }, 400)
    const trackMock = vi.fn()
    window.umami = { track: trackMock }

    await expect(request('/api/v1/convert/raster', { method: 'POST' })).rejects.toThrow()

    expect(trackMock).toHaveBeenCalledTimes(1)
    const [, data] = trackMock.mock.calls[0] as [string, Record<string, unknown>]
    expect(data.success).toBe(false)
    expect(data.method).toBe('POST')
  })

  it('does not throw when window.umami is undefined', async () => {
    mockFetch({ ok: true, data: [] })
    window.umami = undefined

    await expect(request('/api/v1/databases')).resolves.toEqual([])
  })
})
