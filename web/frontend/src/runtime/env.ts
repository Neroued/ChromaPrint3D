import { getElectronApi, isElectronRuntime } from './platform'

const DEFAULT_ELECTRON_API_BASE = 'http://127.0.0.1:18080'
const DEFAULT_ICP_RECORD_URL = 'https://beian.miit.gov.cn/'
const DEFAULT_PUBLIC_SECURITY_RECORD_URL = 'https://beian.mps.gov.cn/'
const DEFAULT_UPLOAD_MAX_MB = 50
const DEFAULT_UPLOAD_MAX_PIXELS = 4096 * 4096

export type ApiBaseSource = 'electron-env' | 'vite-env' | 'electron-default' | 'same-origin'

type ApiBaseResolution = {
  base: string
  source: ApiBaseSource
}

function normalizeBase(base: string): string {
  return base.replace(/\/+$/, '')
}

function normalizeText(raw: string | undefined): string {
  return raw?.trim() ?? ''
}

function parsePositiveInteger(raw: unknown): number | null {
  if (typeof raw === 'number') {
    if (Number.isInteger(raw) && raw > 0) return raw
    return null
  }
  if (typeof raw !== 'string') return null
  const normalized = raw.trim()
  if (!normalized) return null
  const parsed = Number(normalized)
  if (!Number.isInteger(parsed) || parsed <= 0) return null
  return parsed
}

function resolveApiBase(): ApiBaseResolution {
  const electronBase = getElectronApi()?.env?.apiBase
  const envBase = import.meta.env.VITE_API_BASE

  if (electronBase?.trim()) {
    return {
      base: normalizeBase(electronBase.trim()),
      source: 'electron-env',
    }
  }
  if (envBase?.trim()) {
    return {
      base: normalizeBase(envBase.trim()),
      source: 'vite-env',
    }
  }
  if (isElectronRuntime()) {
    return {
      base: DEFAULT_ELECTRON_API_BASE,
      source: 'electron-default',
    }
  }
  return {
    base: '',
    source: 'same-origin',
  }
}

export function getApiBase(): string {
  return resolveApiBase().base
}

export function getApiBaseSource(): ApiBaseSource {
  return resolveApiBase().source
}

export function buildApiUrl(path: string): string {
  return `${getApiBase()}${path}`
}

export function getUploadMaxMb(): number {
  const electronMaxUploadMb = parsePositiveInteger(getElectronApi()?.env?.uploadMaxMb)
  if (electronMaxUploadMb !== null) return electronMaxUploadMb
  const envMaxUploadMb = parsePositiveInteger(import.meta.env.VITE_UPLOAD_MAX_MB)
  if (envMaxUploadMb !== null) return envMaxUploadMb
  return DEFAULT_UPLOAD_MAX_MB
}

export function getUploadMaxBytes(): number {
  return getUploadMaxMb() * 1024 * 1024
}

export function getUploadMaxPixels(): number {
  const electronMaxPixels = parsePositiveInteger(getElectronApi()?.env?.uploadMaxPixels)
  if (electronMaxPixels !== null) return electronMaxPixels
  const envMaxPixels = parsePositiveInteger(import.meta.env.VITE_UPLOAD_MAX_PIXELS)
  if (envMaxPixels !== null) return envMaxPixels
  return DEFAULT_UPLOAD_MAX_PIXELS
}

export function getSiteIcpNumber(): string {
  return normalizeText(import.meta.env.VITE_SITE_ICP_NUMBER)
}

export function getSiteIcpUrl(): string {
  const customUrl = normalizeText(import.meta.env.VITE_SITE_ICP_URL)
  return customUrl || DEFAULT_ICP_RECORD_URL
}

export function getSitePublicSecurityRecordNumber(): string {
  return normalizeText(import.meta.env.VITE_SITE_PUBLIC_SECURITY_RECORD_NUMBER)
}

export function getSitePublicSecurityRecordUrl(): string {
  const customUrl = normalizeText(import.meta.env.VITE_SITE_PUBLIC_SECURITY_RECORD_URL)
  return customUrl || DEFAULT_PUBLIC_SECURITY_RECORD_URL
}

export function getStableUrl(): string {
  return normalizeText(import.meta.env.VITE_STABLE_URL)
}

export function getPreviewUrl(): string {
  return normalizeText(import.meta.env.VITE_PREVIEW_URL)
}
