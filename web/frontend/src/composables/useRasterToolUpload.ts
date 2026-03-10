import { computed, ref, watch } from 'vue'
import type { UploadFileInfo } from 'naive-ui'
import { useObjectUrlLifecycle } from './useObjectUrlLifecycle'
import {
  RASTER_IMAGE_ACCEPT,
  RASTER_IMAGE_FORMATS_TEXT,
  validateImageUploadFile,
} from '../domain/upload/imageUploadValidation'
import { getUploadMaxMb, getUploadMaxPixels } from '../runtime/env'

type UseRasterToolUploadOptions = {
  onReset: () => void
  onError: (message: string | null) => void
}

export function useRasterToolUpload(options: UseRasterToolUploadOptions) {
  const file = ref<File | null>(null)
  const fileList = ref<UploadFileInfo[]>([])
  const originalUrl = ref<string | null>(null)
  const { createUrl, revokeUrl } = useObjectUrlLifecycle()

  const backendMaxUploadMb = getUploadMaxMb()
  const maxPixelText = getUploadMaxPixels().toLocaleString()

  const imageInfo = computed(() => {
    if (!file.value) return null
    const sizeMB = (file.value.size / 1024 / 1024).toFixed(2)
    return `${file.value.name} (${sizeMB} MB)`
  })

  async function handleUploadChange({ fileList: nextFileList }: { fileList: UploadFileInfo[] }) {
    if (nextFileList.length === 0) {
      clearFile()
      return
    }

    const latest = nextFileList[nextFileList.length - 1]
    if (!latest?.file) return

    const validation = await validateImageUploadFile(latest.file, 'raster-tool')
    if (!validation.ok) {
      clearFile()
      options.onError(validation.message)
      return
    }

    options.onError(null)
    file.value = latest.file
  }

  function clearFile() {
    file.value = null
    fileList.value = []
    revokeUrl(originalUrl.value)
    originalUrl.value = null
    options.onReset()
  }

  watch(file, (nextFile) => {
    revokeUrl(originalUrl.value)
    originalUrl.value = nextFile ? createUrl(nextFile) : null
    options.onReset()
  })

  return {
    backendMaxUploadMb,
    maxPixelText,
    file,
    fileList,
    imageInfo,
    originalUrl,
    clearFile,
    handleUploadChange,
    rasterImageAccept: RASTER_IMAGE_ACCEPT,
    rasterImageFormatsText: RASTER_IMAGE_FORMATS_TEXT,
    revokeOriginalUrl: () => revokeUrl(originalUrl.value),
  }
}
