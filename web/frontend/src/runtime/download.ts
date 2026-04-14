import { getElectronApi } from './platform'
import { createBlobUrl, revokeBlobUrl } from './blob'
import { fetchBlobWithSession } from './protectedRequest'

function clickDownloadLink(url: string, filename: string): void {
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

export async function openExternalUrl(url: string): Promise<void> {
  const electronDownload = getElectronApi()?.download
  if (electronDownload?.openExternal) {
    await electronDownload.openExternal(url)
    return
  }
  window.open(url, '_blank')
}

export async function downloadFromUrl(url: string, filename: string): Promise<void> {
  const blob = await fetchBlobWithSession(url)
  const blobUrl = createBlobUrl(blob)
  try {
    const electronDownload = getElectronApi()?.download
    if (electronDownload?.saveObjectUrlAs) {
      await electronDownload.saveObjectUrlAs(blobUrl, filename)
    } else {
      clickDownloadLink(blobUrl, filename)
    }
  } finally {
    revokeBlobUrl(blobUrl)
  }
}

export async function downloadObjectUrl(url: string, filename: string): Promise<void> {
  const electronDownload = getElectronApi()?.download
  if (electronDownload?.saveObjectUrlAs) {
    await electronDownload.saveObjectUrlAs(url, filename)
    return
  }
  clickDownloadLink(url, filename)
}
