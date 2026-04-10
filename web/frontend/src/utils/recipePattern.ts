import type { PaletteChannel } from '../types'

/**
 * Build a mapping from uppercase first-letter to palette channel indices.
 * Multiple channels may share the same letter (e.g. "White" and "Warm White" → 'W').
 */
export function buildPaletteLetterMap(palette: PaletteChannel[]): Map<string, number[]> {
  const map = new Map<string, number[]>()
  for (let i = 0; i < palette.length; i++) {
    const name = palette[i]?.color
    if (!name) continue
    const letter = name.charAt(0).toUpperCase()
    const arr = map.get(letter)
    if (arr) {
      arr.push(i)
    } else {
      map.set(letter, [i])
    }
  }
  return map
}

/**
 * Build a human-readable palette hint string for tooltips (e.g. "W=White B=Black").
 */
export function buildPaletteHint(palette: PaletteChannel[]): string {
  const seen = new Set<string>()
  const parts: string[] = []
  for (const ch of palette) {
    if (!ch?.color) continue
    const letter = ch.color.charAt(0).toUpperCase()
    if (seen.has(letter)) continue
    seen.add(letter)
    parts.push(`${letter}=${ch.color}`)
  }
  return parts.join('  ')
}

/**
 * Validate a recipe pattern string against the given palette.
 * Returns null if valid, or an error message describing the first invalid character.
 */
export function validateRecipePattern(pattern: string, palette: PaletteChannel[]): string | null {
  if (!pattern) return null
  const letterMap = buildPaletteLetterMap(palette)
  for (const ch of pattern) {
    if (ch === '-' || ch === '?' || ch === '*') continue
    if (/[a-zA-Z]/.test(ch)) {
      if (!letterMap.has(ch.toUpperCase())) {
        return ch
      }
      continue
    }
    return ch
  }
  return null
}
