import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const electronRuntime = vi.hoisted(() => ({
  api: undefined as
    | { env?: { uploadMaxMb?: number | string; uploadMaxPixels?: number | string } }
    | undefined,
}))

vi.mock('./platform', () => ({
  getElectronApi: () => electronRuntime.api,
  isElectronRuntime: () => false,
}))

import { getUploadMaxBytes, getUploadMaxMb, getUploadMaxPixels } from './env'

describe('runtime upload limits', () => {
  beforeEach(() => {
    electronRuntime.api = undefined
    vi.unstubAllEnvs()
  })

  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('在无配置时回退默认值', () => {
    expect(getUploadMaxMb()).toBe(50)
    expect(getUploadMaxBytes()).toBe(50 * 1024 * 1024)
    expect(getUploadMaxPixels()).toBe(4096 * 4096)
  })

  it('使用 VITE 环境变量作为 Web 配置', () => {
    vi.stubEnv('VITE_UPLOAD_MAX_MB', '30')
    vi.stubEnv('VITE_UPLOAD_MAX_PIXELS', '12000000')
    expect(getUploadMaxMb()).toBe(30)
    expect(getUploadMaxBytes()).toBe(30 * 1024 * 1024)
    expect(getUploadMaxPixels()).toBe(12_000_000)
  })

  it('忽略无效 VITE 上传配置并回退默认值', () => {
    vi.stubEnv('VITE_UPLOAD_MAX_MB', '-1')
    vi.stubEnv('VITE_UPLOAD_MAX_PIXELS', 'invalid')
    expect(getUploadMaxMb()).toBe(50)
    expect(getUploadMaxPixels()).toBe(4096 * 4096)
  })

  it('Electron 配置优先于 VITE 环境变量', () => {
    vi.stubEnv('VITE_UPLOAD_MAX_MB', '20')
    vi.stubEnv('VITE_UPLOAD_MAX_PIXELS', '8000000')
    electronRuntime.api = {
      env: {
        uploadMaxMb: 120,
        uploadMaxPixels: 25_000_000,
      },
    }
    expect(getUploadMaxMb()).toBe(120)
    expect(getUploadMaxPixels()).toBe(25_000_000)
  })

  it('Electron 配置无效时回退到 VITE 配置', () => {
    vi.stubEnv('VITE_UPLOAD_MAX_MB', '18')
    vi.stubEnv('VITE_UPLOAD_MAX_PIXELS', '7000000')
    electronRuntime.api = {
      env: {
        uploadMaxMb: '0',
        uploadMaxPixels: '-1',
      },
    }
    expect(getUploadMaxMb()).toBe(18)
    expect(getUploadMaxPixels()).toBe(7_000_000)
  })
})
