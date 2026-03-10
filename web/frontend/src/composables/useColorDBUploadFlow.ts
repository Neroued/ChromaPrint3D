import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import type { UploadFileInfo } from 'naive-ui'
import { uploadColorDB } from '../api/session'
import type { ColorDBInfo } from '../types'

export interface UploadResultItem {
  fileName: string
  success: boolean
  dbName?: string
  error?: string
}

type UseColorDBUploadFlowOptions = {
  onUploaded?: (result: ColorDBInfo) => void | Promise<void>
  onBatchDone?: (results: UploadResultItem[]) => void | Promise<void>
}

export function useColorDBUploadFlow(options: UseColorDBUploadFlowOptions = {}) {
  const { t } = useI18n()
  const fileList = ref<UploadFileInfo[]>([])
  const uploading = ref(false)
  const uploadError = ref<string | null>(null)
  const batchResults = ref<UploadResultItem[]>([])

  const isBatch = computed(() => fileList.value.length > 1)
  const canUpload = computed(() => fileList.value.length > 0)

  function handleUploadFileChange(data: { fileList: UploadFileInfo[] }) {
    fileList.value = data.fileList
    uploadError.value = null
    batchResults.value = []
    return false
  }

  async function handleUpload(): Promise<ColorDBInfo[] | null> {
    if (!canUpload.value || fileList.value.length === 0) return null

    uploading.value = true
    uploadError.value = null
    batchResults.value = []

    const files = fileList.value.map((f) => f.file).filter(Boolean) as File[]
    if (files.length === 0) {
      uploading.value = false
      return null
    }

    const results: UploadResultItem[] = []
    const uploaded: ColorDBInfo[] = []

    for (const file of files) {
      try {
        const result = await uploadColorDB(file)
        results.push({ fileName: file.name, success: true, dbName: result.name })
        uploaded.push(result)
        if (options.onUploaded) {
          await options.onUploaded(result)
        }
      } catch (error: unknown) {
        const msg = error instanceof Error ? error.message : String(error)
        results.push({ fileName: file.name, success: false, error: msg })
      }
    }

    batchResults.value = results

    if (uploaded.length > 0) {
      fileList.value = []
    }

    if (uploaded.length === 0) {
      uploadError.value =
        files.length === 1
          ? (results[0]?.error ?? t('colordb.upload.messages.singleUploadFailed'))
          : t('colordb.upload.messages.allUploadFailed')
    }

    if (options.onBatchDone) {
      await options.onBatchDone(results)
    }

    uploading.value = false
    return uploaded.length > 0 ? uploaded : null
  }

  function clearFiles() {
    fileList.value = []
    uploadError.value = null
    batchResults.value = []
  }

  return {
    fileList,
    uploading,
    uploadError,
    canUpload,
    isBatch,
    batchResults,
    handleUploadFileChange,
    handleUpload,
    clearFiles,
  }
}
