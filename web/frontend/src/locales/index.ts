import { createI18n } from 'vue-i18n'
import zhCN from './zh-CN'
import en from './en'

export type MessageSchema = typeof zhCN
export type SupportedLocale = 'zh-CN' | 'en'

export const LOCALE_STORAGE_KEY = 'chromaprint3d-locale'
export const SUPPORTED_LOCALES: SupportedLocale[] = ['zh-CN', 'en']

function detectInitialLocale(): SupportedLocale {
  if (typeof localStorage !== 'undefined') {
    const stored = localStorage.getItem(LOCALE_STORAGE_KEY)
    if (stored === 'zh-CN' || stored === 'en') return stored
  }
  if (typeof navigator !== 'undefined') {
    return navigator.language.startsWith('zh') ? 'zh-CN' : 'en'
  }
  return 'zh-CN'
}

const i18n = createI18n<[MessageSchema], SupportedLocale>({
  legacy: false,
  locale: detectInitialLocale(),
  fallbackLocale: 'en',
  messages: {
    'zh-CN': zhCN,
    en,
  },
})

export default i18n
