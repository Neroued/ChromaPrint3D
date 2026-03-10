import { downloadFromUrl, downloadObjectUrl } from '../runtime/download'
import i18n from '../locales'

function toMessage(err: unknown, fallback: string): string {
  if (err instanceof Error && err.message) return err.message
  return fallback
}

export function useBlobDownload(onError?: (message: string) => void) {
  async function downloadByUrl(url: string, filename: string): Promise<void> {
    try {
      await downloadFromUrl(url, filename)
    } catch (err) {
      onError?.(toMessage(err, i18n.global.t('common.downloadFailed')))
      throw err
    }
  }

  async function downloadByObjectUrl(url: string, filename: string): Promise<void> {
    try {
      await downloadObjectUrl(url, filename)
    } catch (err) {
      onError?.(toMessage(err, i18n.global.t('common.downloadFailed')))
      throw err
    }
  }

  return {
    downloadByUrl,
    downloadByObjectUrl,
  }
}
