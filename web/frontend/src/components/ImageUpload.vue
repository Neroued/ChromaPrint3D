<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { NUpload, NUploadDragger, NCard, NText, NSpace, NButton, NTag, NAlert } from 'naive-ui'
import type { UploadFileInfo } from 'naive-ui'
import { useObjectUrlLifecycle } from '../composables/useObjectUrlLifecycle'
import { usePanZoom } from '../composables/usePanZoom'
import ZoomableImageViewport from './common/ZoomableImageViewport.vue'
import type { InputType } from '../types'
import { useAppStore } from '../stores/app'
import {
  CONVERT_IMAGE_ACCEPT,
  CONVERT_IMAGE_FORMATS_TEXT,
  readRasterImageDimensions,
  validateImageUploadFile,
} from '../domain/upload/imageUploadValidation'
import { getUploadMaxMb, getUploadMaxPixels } from '../runtime/env'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

defineProps<{
  disabled?: boolean
}>()

const fileList = ref<UploadFileInfo[]>([])
const previewUrl = ref<string | null>(null)
const dimensions = ref<{ width: number; height: number } | null>(null)
const detectedType = ref<InputType>('raster')
const uploadError = ref<string | null>(null)
let dimensionsTaskVersion = 0
const { createUrl, revokeUrl } = useObjectUrlLifecycle()
const appStore = useAppStore()
const { selectedFile } = storeToRefs(appStore)

const previewPanZoom = usePanZoom()

function isSvgFile(file: File): boolean {
  return file.type === 'image/svg+xml' || file.name.toLowerCase().endsWith('.svg')
}

const backendMaxUploadMb = getUploadMaxMb()
const maxPixelText = getUploadMaxPixels().toLocaleString()

const fileInfo = computed(() => {
  const file = selectedFile.value
  if (!file) return null
  const sizeMB = (file.size / 1024 / 1024).toFixed(2)
  const dimStr = dimensions.value ? ` | ${dimensions.value.width}×${dimensions.value.height}` : ''
  return `${file.name} (${sizeMB} MB${dimStr})`
})

async function handleChange(options: { fileList: UploadFileInfo[] }) {
  const files = options.fileList
  if (files.length === 0) {
    appStore.setSelectedFile(null)
    appStore.setImageDimensions(null)
    appStore.setInputType('raster')
    dimensions.value = null
    detectedType.value = 'raster'
    uploadError.value = null
    revokeUrl(previewUrl.value)
    previewUrl.value = null
    return
  }
  const latest = files[files.length - 1]
  if (!latest) return
  const rawFile = latest.file
  if (rawFile) {
    const validation = await validateImageUploadFile(rawFile, 'convert')
    if (!validation.ok) {
      clearFile()
      uploadError.value = validation.message
      return
    }
    uploadError.value = null
    appStore.setSelectedFile(rawFile)
  }
}

function clearFile() {
  dimensionsTaskVersion += 1
  fileList.value = []
  dimensions.value = null
  detectedType.value = 'raster'
  uploadError.value = null
  appStore.setSelectedFile(null)
  appStore.setImageDimensions(null)
  appStore.setInputType('raster')
  revokeUrl(previewUrl.value)
  previewUrl.value = null
  previewPanZoom.resetView()
}

async function syncImageDimensions(file: File) {
  const currentVersion = ++dimensionsTaskVersion
  const dims = await readRasterImageDimensions(file)
  if (currentVersion !== dimensionsTaskVersion) return
  dimensions.value = dims
  appStore.setImageDimensions(dims)
}

watch(
  () => selectedFile.value,
  (newFile) => {
    dimensionsTaskVersion += 1
    revokeUrl(previewUrl.value)
    previewUrl.value = null
    previewPanZoom.resetView()
    if (newFile) {
      uploadError.value = null
      const type: InputType = isSvgFile(newFile) ? 'vector' : 'raster'
      detectedType.value = type
      appStore.setInputType(type)

      previewUrl.value = createUrl(newFile)
      if (type === 'raster') {
        void syncImageDimensions(newFile)
      } else {
        dimensions.value = null
        appStore.setImageDimensions(null)
      }
    } else {
      fileList.value = []
      detectedType.value = 'raster'
      appStore.setImageDimensions(null)
      appStore.setInputType('raster')
    }
  },
)
</script>

<template>
  <NCard :title="t('imageUpload.title')" size="small">
    <NUpload
      v-if="!selectedFile"
      :accept="CONVERT_IMAGE_ACCEPT"
      :max="1"
      :default-upload="false"
      :disabled="disabled"
      :file-list="fileList"
      :show-file-list="false"
      @update:file-list="
        (v: UploadFileInfo[]) => {
          fileList = v
        }
      "
      @change="handleChange"
    >
      <NUploadDragger>
        <NSpace vertical align="center" justify="center" style="padding: 32px 16px">
          <NText depth="3" style="font-size: 14px"> {{ t('imageUpload.dropHint') }} </NText>
          <NText depth="3" style="font-size: 12px">
            {{ t('imageUpload.formatHint', { formats: CONVERT_IMAGE_FORMATS_TEXT }) }}
          </NText>
          <NText depth="3" style="font-size: 11px">
            {{ t('imageUpload.sizeHint', { maxMb: backendMaxUploadMb, maxPixels: maxPixelText }) }}
          </NText>
        </NSpace>
      </NUploadDragger>
    </NUpload>
    <NAlert v-if="uploadError" type="error" style="margin-bottom: 8px">
      {{ uploadError }}
    </NAlert>

    <div v-if="selectedFile">
      <div class="upload-preview-header">
        <NSpace align="center" :size="8">
          <NText depth="3" style="font-size: 12px">
            {{ fileInfo }}
          </NText>
          <NTag
            size="tiny"
            :type="detectedType === 'vector' ? 'success' : 'info'"
            :bordered="false"
          >
            {{
              detectedType === 'vector' ? t('imageUpload.vectorType') : t('imageUpload.rasterType')
            }}
          </NTag>
        </NSpace>
        <NSpace :size="4" align="center">
          <NButton size="tiny" quaternary @click="previewPanZoom.resetView">
            {{ t('imageUpload.resetView') }}
          </NButton>
          <NButton size="tiny" quaternary type="error" :disabled="disabled" @click="clearFile">
            {{ t('imageUpload.removeFile') }}
          </NButton>
        </NSpace>
      </div>
      <NText depth="3" style="font-size: 11px; display: block; margin-bottom: 4px">
        {{ t('imageUpload.zoomHint') }}
      </NText>
      <ZoomableImageViewport
        :src="previewUrl ?? undefined"
        alt="preview"
        :height="300"
        :controller="previewPanZoom"
      />
    </div>
  </NCard>
</template>

<style scoped>
.upload-preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
}
</style>
