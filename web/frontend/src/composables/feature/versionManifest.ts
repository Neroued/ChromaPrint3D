export type VersionManifest = {
  version: string
  download_url: string
  changelog: string
  changelog_en?: string
  published_at: string
}

const DEFAULT_TIMEOUT_MS = 5000

export function isValidManifest(data: unknown): data is VersionManifest {
  if (typeof data !== 'object' || data === null) return false
  const obj = data as Record<string, unknown>
  return typeof obj.version === 'string' && /^\d+\.\d+\.\d+/.test(obj.version)
}

export function compareVersions(a: string, b: string): number {
  const pa = a.split('.').map(Number)
  const pb = b.split('.').map(Number)
  const len = Math.max(pa.length, pb.length)
  for (let i = 0; i < len; i++) {
    const na = pa[i] ?? 0
    const nb = pb[i] ?? 0
    if (na !== nb) return na - nb
  }
  return 0
}

export async function fetchManifestFromUrl(
  url: string,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<VersionManifest | null> {
  if (!url) return null

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const resp = await fetch(url, { signal: controller.signal, cache: 'no-cache' })
    if (!resp.ok) return null
    const data: unknown = await resp.json()
    if (!isValidManifest(data)) return null
    return data
  } catch {
    return null
  } finally {
    clearTimeout(timer)
  }
}
