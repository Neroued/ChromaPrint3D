import i18n from './locales'
import type { WritableComputedRef } from 'vue'
import type { SupportedLocale } from './locales'
;(i18n.global.locale as unknown as WritableComputedRef<SupportedLocale>).value = 'zh-CN'
