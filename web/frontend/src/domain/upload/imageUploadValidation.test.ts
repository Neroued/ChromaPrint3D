import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
const runtimeLimits = vi.hoisted(() => ({
  uploadMaxMb: 50,
  maxPixels: 4096 * 4096,
}))

vi.mock('../../runtime/env', () => ({
  getUploadMaxMb: () => runtimeLimits.uploadMaxMb,
  getUploadMaxBytes: () => runtimeLimits.uploadMaxMb * 1024 * 1024,
  getUploadMaxPixels: () => runtimeLimits.maxPixels,
}))

import { validateImageUploadFile, type UploadValidationResult } from './imageUploadValidation'

type MockImageBehavior = { kind: 'load'; width: number; height: number } | { kind: 'error' }

class MockImage {
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  naturalWidth = 0
  naturalHeight = 0

  set src(_value: string) {
    const next = mockImageQueue.shift()
    queueMicrotask(() => {
      if (!next || next.kind === 'error') {
        this.onerror?.()
        return
      }
      this.naturalWidth = next.width
      this.naturalHeight = next.height
      this.onload?.()
    })
  }
}

const mockImageQueue: MockImageBehavior[] = []
const originalImage = globalThis.Image
const originalCreateObjectUrl = URL.createObjectURL
const originalRevokeObjectUrl = URL.revokeObjectURL

function queueImageLoad(width: number, height: number) {
  mockImageQueue.push({ kind: 'load', width, height })
}

function queueImageError() {
  mockImageQueue.push({ kind: 'error' })
}

async function expectFailed(resultPromise: Promise<UploadValidationResult>) {
  const result = await resultPromise
  expect(result.ok).toBe(false)
  if (!result.ok) return result.message
  return ''
}

describe('imageUploadValidation', () => {
  beforeEach(() => {
    mockImageQueue.length = 0
    runtimeLimits.uploadMaxMb = 50
    runtimeLimits.maxPixels = 4096 * 4096
    Object.defineProperty(globalThis, 'Image', {
      configurable: true,
      writable: true,
      value: MockImage,
    })
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      writable: true,
      value: vi.fn(() => 'blob:mock'),
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      writable: true,
      value: vi.fn(),
    })
  })

  afterEach(() => {
    Object.defineProperty(globalThis, 'Image', {
      configurable: true,
      writable: true,
      value: originalImage,
    })
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      writable: true,
      value: originalCreateObjectUrl,
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      writable: true,
      value: originalRevokeObjectUrl,
    })
    vi.restoreAllMocks()
  })

  it('允许转换页面上传 SVG', async () => {
    const file = new File(['<svg></svg>'], 'icon.svg', { type: 'image/svg+xml' })
    const result = await validateImageUploadFile(file, 'convert', {
      maxUploadBytes: 1024,
      maxPixels: 1000,
    })
    expect(result).toEqual({ ok: true, inputType: 'vector', dimensions: null })
  })

  it('拒绝抠图/矢量化页面上传 SVG', async () => {
    const file = new File(['<svg></svg>'], 'icon.svg', { type: 'image/svg+xml' })
    const message = await expectFailed(
      validateImageUploadFile(file, 'raster-tool', {
        maxUploadBytes: 1024,
        maxPixels: 1000,
      }),
    )
    expect(message).toContain('仅支持位图格式')
  })

  it('拒绝不支持的位图格式', async () => {
    const file = new File(['x'], 'demo.webp', { type: 'image/webp' })
    const message = await expectFailed(
      validateImageUploadFile(file, 'convert', {
        maxUploadBytes: 1024,
        maxPixels: 1000,
      }),
    )
    expect(message).toContain('不支持该图片格式')
  })

  it('拒绝超过上传体积上限的文件', async () => {
    const file = new File(['abcdef'], 'demo.png', { type: 'image/png' })
    const message = await expectFailed(
      validateImageUploadFile(file, 'convert', {
        maxUploadBytes: 2,
        maxPixels: 10000,
      }),
    )
    expect(message).toContain('文件大小超过后端上传限制')
  })

  it('默认读取 runtime 上传体积上限', async () => {
    runtimeLimits.uploadMaxMb = 1
    const file = new File([new Uint8Array(1024 * 1024 + 1)], 'demo.png', { type: 'image/png' })
    const message = await expectFailed(validateImageUploadFile(file, 'convert'))
    expect(message).toContain('文件大小超过后端上传限制（1MB）')
  })

  it('拒绝超过像素上限的位图', async () => {
    queueImageLoad(200, 200)
    const file = new File(['x'], 'demo.png', { type: 'image/png' })
    const message = await expectFailed(
      validateImageUploadFile(file, 'convert', {
        maxUploadBytes: 1024,
        maxPixels: 10000,
      }),
    )
    expect(message).toContain('图片像素超过后端上限')
  })

  it('默认读取 runtime 像素上限', async () => {
    runtimeLimits.uploadMaxMb = 4
    runtimeLimits.maxPixels = 10_000
    queueImageLoad(101, 100)
    const file = new File(['x'], 'demo.png', { type: 'image/png' })
    const message = await expectFailed(validateImageUploadFile(file, 'convert'))
    expect(message).toContain('图片像素超过后端上限')
  })

  it('拒绝浏览器无法解码的位图', async () => {
    queueImageError()
    const file = new File(['x'], 'demo.png', { type: 'image/png' })
    const message = await expectFailed(
      validateImageUploadFile(file, 'convert', {
        maxUploadBytes: 1024,
        maxPixels: 10000,
      }),
    )
    expect(message).toContain('图片无法解码')
  })

  it('调用方覆盖值优先于 runtime 默认值', async () => {
    runtimeLimits.uploadMaxMb = 1
    runtimeLimits.maxPixels = 10_000
    queueImageLoad(200, 200)
    const file = new File(['abcdef'], 'demo.jpg', { type: 'image/jpeg' })
    const result = await validateImageUploadFile(file, 'raster-tool', {
      maxUploadBytes: 1024 * 1024 * 2,
      maxPixels: 50_000,
    })
    expect(result).toEqual({
      ok: true,
      inputType: 'raster',
      dimensions: { width: 200, height: 200 },
    })
  })

  it('允许通过体积和像素校验的位图', async () => {
    queueImageLoad(80, 120)
    const file = new File(['abcdef'], 'demo.jpg', { type: 'image/jpeg' })
    const result = await validateImageUploadFile(file, 'raster-tool', {
      maxUploadBytes: 1024,
      maxPixels: 10000,
    })
    expect(result).toEqual({
      ok: true,
      inputType: 'raster',
      dimensions: { width: 80, height: 120 },
    })
  })
})
