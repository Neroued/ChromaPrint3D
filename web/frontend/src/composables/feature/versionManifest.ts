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
  return typeof obj.version === 'string' && /^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$/.test(obj.version)
}

function parseVersion(v: string): { parts: number[]; prerelease: number | null } {
  const hyphen = v.indexOf('-')
  const base = hyphen === -1 ? v : v.slice(0, hyphen)
  const parts = base.split('.').map(Number)
  if (hyphen === -1) return { parts, prerelease: null }
  const pre = v.slice(hyphen + 1)
  const m = /^rc\.(\d+)$/.exec(pre)
  return { parts, prerelease: m ? Number(m[1]) : 0 }
}

/**
 * Semver-aware comparison with pre-release support.
 * Ordering: 1.2.12 > 1.2.12-rc.2 > 1.2.12-rc.1 > 1.2.11
 */
export function compareVersions(a: string, b: string): number {
  const va = parseVersion(a)
  const vb = parseVersion(b)
  const len = Math.max(va.parts.length, vb.parts.length)
  for (let i = 0; i < len; i++) {
    const na = va.parts[i] ?? 0
    const nb = vb.parts[i] ?? 0
    if (na !== nb) return na - nb
  }
  if (va.prerelease === null && vb.prerelease === null) return 0
  if (va.prerelease === null) return 1
  if (vb.prerelease === null) return -1
  return va.prerelease - vb.prerelease
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
