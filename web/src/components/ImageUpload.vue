<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { NUpload, NUploadDragger, NCard, NImage, NText, NSpace, NButton } from 'naive-ui'
import type { UploadFileInfo } from 'naive-ui'

const props = defineProps<{
  modelValue: File | null
  disabled?: boolean
}>()

const emit = defineEmits<{
  'update:modelValue': [file: File | null]
}>()

const fileList = ref<UploadFileInfo[]>([])
const previewUrl = ref<string | null>(null)

const imageInfo = computed(() => {
  const file = props.modelValue
  if (!file) return null
  const sizeMB = (file.size / 1024 / 1024).toFixed(2)
  return `${file.name} (${sizeMB} MB)`
})

function handleChange(options: { fileList: UploadFileInfo[] }) {
  const files = options.fileList
  if (files.length === 0) {
    emit('update:modelValue', null)
    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value)
      previewUrl.value = null
    }
    return
  }
  const latest = files[files.length - 1]
  if (!latest) return
  const rawFile = latest.file
  if (rawFile) {
    emit('update:modelValue', rawFile)
  }
}

function clearImage() {
  fileList.value = []
  emit('update:modelValue', null)
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
}

watch(
  () => props.modelValue,
  (newFile) => {
    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value)
      previewUrl.value = null
    }
    if (newFile) {
      previewUrl.value = URL.createObjectURL(newFile)
    } else {
      fileList.value = []
    }
  },
)
</script>

<template>
  <NCard title="图片上传" size="small">
    <!-- Upload area: shown when no image -->
    <NUpload
      v-if="!modelValue"
      accept="image/*"
      :max="1"
      :default-upload="false"
      :disabled="disabled"
      :file-list="fileList"
      :show-file-list="false"
      @update:file-list="(v: UploadFileInfo[]) => { fileList = v }"
      @change="handleChange"
    >
      <NUploadDragger>
        <NSpace vertical align="center" justify="center" style="padding: 32px 16px">
          <NText depth="3" style="font-size: 14px">
            点击或拖拽图片到此处上传
          </NText>
          <NText depth="3" style="font-size: 12px">
            支持 JPG / PNG / BMP / TIFF 格式
          </NText>
        </NSpace>
      </NUploadDragger>
    </NUpload>

    <!-- Preview area: shown when image is uploaded -->
    <div v-else class="preview-container">
      <div class="preview-header">
        <NText depth="3" style="font-size: 12px">
          {{ imageInfo }}
        </NText>
        <NButton size="tiny" quaternary type="error" :disabled="disabled" @click="clearImage">
          移除图片
        </NButton>
      </div>
      <NImage
        :src="previewUrl ?? undefined"
        object-fit="contain"
        :img-props="{ style: 'max-width: 100%; max-height: 400px; object-fit: contain; cursor: zoom-in;' }"
        style="border-radius: 4px"
      />
      <NText depth="3" style="font-size: 11px; margin-top: 4px; display: block">
        点击图片可放大查看，支持滚轮缩放和拖拽移动
      </NText>
    </div>
  </NCard>
</template>

<style scoped>
.preview-container {
  text-align: center;
}

.preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
</style>
