import type { Composer } from 'vue-i18n'

interface ErrorPattern {
  pattern: RegExp
  key: string
}

const ERROR_PATTERNS: ErrorPattern[] = [
  {
    pattern: /No compatible ColorDB entries for color_layers=\d+/,
    key: 'taskErrors.noCompatibleDbEntries',
  },
  {
    pattern: /color_layers must be positive/,
    key: 'taskErrors.colorLayersMustBePositive',
  },
  {
    pattern: /color_layers must be between/,
    key: 'taskErrors.colorLayersOutOfRange',
  },
  {
    pattern: /Model-only matching requires a compatible model package/,
    key: 'taskErrors.modelOnlyRequiresPackage',
  },
  {
    pattern: /palette is empty/,
    key: 'taskErrors.paletteEmpty',
  },
]

export function mapTaskError(raw: string, t: Composer['t']): string {
  for (const { pattern, key } of ERROR_PATTERNS) {
    if (pattern.test(raw)) {
      const translated = t(key)
      if (translated !== key) return translated
    }
  }
  return raw
}
