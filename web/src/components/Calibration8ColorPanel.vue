<script setup lang="ts">
import { ref, computed } from 'vue'
import {
  NCard,
  NButton,
  NInput,
  NSpace,
  NSteps,
  NStep,
  NAlert,
  NUpload,
  NSpin,
  NTag,
  NDivider,
  NGrid,
  NGi,
  useMessage,
  type UploadFileInfo,
} from 'naive-ui'
import {
  generate8ColorBoard,
  getBoardModelUrl,
  getBoardMetaUrl,
  buildColorDB,
  getSessionColorDBDownloadUrl,
  uploadColorDB,
} from '../api'
import type { PaletteChannel } from '../types'

const message = useMessage()

const emit = defineEmits<{
  (e: 'colordb-updated'): void
}>()

// ======== Section 1: Generate calibration boards ========

const DEFAULT_8_COLORS: PaletteChannel[] = [
  { color: 'Bamboo Green', material: 'PLA Basic' },
  { color: 'Black', material: 'PLA Basic' },
  { color: 'Blue', material: 'PLA Basic' },
  { color: 'Cyan', material: 'PLA Basic' },
  { color: 'Magenta', material: 'PLA Basic' },
  { color: 'Red', material: 'PLA Basic' },
  { color: 'White', material: 'PLA Basic' },
  { color: 'Yellow', material: 'PLA Basic' },
]

const palette = ref<PaletteChannel[]>(
  DEFAULT_8_COLORS.map(c => ({ ...c })),
)

const board1Id = ref<string | null>(null)
const board2Id = ref<string | null>(null)
const generating1 = ref(false)
const generating2 = ref(false)

function validatePalette(): boolean {
  for (const ch of palette.value) {
    if (!ch.color.trim()) {
      message.error('请填写所有颜色名称')
      return false
    }
  }
  return true
}

async function handleGenerateBoard(boardIndex: number) {
  if (!validatePalette()) return
  const isBoard1 = boardIndex === 1
  const generatingRef = isBoard1 ? generating1 : generating2
  const boardIdRef = isBoard1 ? board1Id : board2Id

  generatingRef.value = true
  boardIdRef.value = null
  try {
    const resp = await generate8ColorBoard({
      palette: palette.value,
      board_index: boardIndex,
    })
    boardIdRef.value = resp.board_id
    message.success(`校准板 ${boardIndex} 生成成功！`)
  } catch (e: unknown) {
    message.error('生成失败: ' + (e instanceof Error ? e.message : String(e)))
  } finally {
    generatingRef.value = false
  }
}

function download3mf(boardId: string | null) {
  if (!boardId) return
  window.open(getBoardModelUrl(boardId), '_blank')
}

function downloadMeta(boardId: string | null) {
  if (!boardId) return
  window.open(getBoardMetaUrl(boardId), '_blank')
}

// ======== Section 3: Build ColorDB ========

const calibImage = ref<File | null>(null)
const calibMeta = ref<File | null>(null)
const dbName = ref('')
const building = ref(false)
const builtDB = ref<{ name: string; num_entries: number; num_channels: number } | null>(null)
const buildError = ref<string | null>(null)

const canBuild = computed(
  () => calibImage.value && calibMeta.value && dbName.value.trim() && /^[a-zA-Z0-9_]+$/.test(dbName.value),
)

function handleImageUpload({ file }: { file: UploadFileInfo }) {
  calibImage.value = file.file ?? null
  return false
}

function handleMetaUpload({ file }: { file: UploadFileInfo }) {
  calibMeta.value = file.file ?? null
  return false
}

async function handleBuildAndNotify() {
  if (!calibImage.value || !calibMeta.value || !dbName.value.trim()) return
  building.value = true
  builtDB.value = null
  buildError.value = null
  try {
    const result = await buildColorDB(calibImage.value, calibMeta.value, dbName.value.trim())
    builtDB.value = {
      name: result.name,
      num_entries: result.num_entries,
      num_channels: result.num_channels,
    }
    message.success(`ColorDB "${result.name}" 构建成功，已自动添加到可用数据库`)
    emit('colordb-updated')
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e)
    buildError.value = msg
    message.error('构建失败')
  } finally {
    building.value = false
  }
}

function downloadBuiltDB() {
  if (!builtDB.value) return
  window.open(getSessionColorDBDownloadUrl(builtDB.value.name), '_blank')
}

// ======== Section 4: Upload ColorDB ========

const uploadFile = ref<File | null>(null)
const uploadName = ref('')
const uploading = ref(false)

function handleUploadFileChange({ file }: { file: UploadFileInfo }) {
  uploadFile.value = file.file ?? null
  if (file.file && !uploadName.value) {
    let name = file.file.name.replace(/\.json$/i, '')
    name = name.replace(/[^a-zA-Z0-9_]/g, '_')
    uploadName.value = name
  }
  return false
}

async function handleUpload() {
  if (!uploadFile.value) return
  uploading.value = true
  try {
    const nameToUse = uploadName.value.trim() || undefined
    const result = await uploadColorDB(uploadFile.value, nameToUse)
    message.success(`ColorDB "${result.name}" 上传成功，已添加到当前会话`)
    emit('colordb-updated')
    uploadFile.value = null
    uploadName.value = ''
  } catch (e: unknown) {
    message.error('上传失败: ' + (e instanceof Error ? e.message : String(e)))
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <NSpace vertical :size="24" style="max-width: 900px; margin: 0 auto">
    <!-- Section 1: Generate Boards -->
    <NCard title="生成八色校准板">
      <NSpace vertical :size="12">
        <NAlert type="info" :bordered="false" style="margin-bottom: 8px">
          八色校准需要两张校准板（各 40x40 = 1600 种配方）。校准板 1 覆盖广泛色域，打印后即可获得良好效果；
          校准板 2 补充更多细节颜色，两张配合使用可获得最佳表现。
        </NAlert>

        <div
          v-for="(ch, idx) in palette"
          :key="idx"
          style="display: flex; gap: 8px; align-items: center"
        >
          <NTag :bordered="false" type="info" size="small">{{ idx + 1 }}</NTag>
          <NInput
            v-model:value="ch.color"
            placeholder="颜色名称（如 White）"
            style="flex: 1"
          />
          <NInput
            v-model:value="ch.material"
            placeholder="材质（如 PLA Basic）"
            style="flex: 1"
          />
        </div>

        <NDivider />

        <!-- Board 1 -->
        <NSpace vertical :size="8">
          <NSpace align="center" :size="8">
            <NButton
              type="primary"
              :loading="generating1"
              @click="handleGenerateBoard(1)"
            >
              生成校准板 1
            </NButton>
            <NTag size="small" type="success" :bordered="false">推荐优先打印</NTag>
          </NSpace>
          <NSpace v-if="board1Id" :size="12">
            <NButton type="success" size="small" @click="download3mf(board1Id)">
              下载 3MF 模型
            </NButton>
            <NButton type="info" size="small" @click="downloadMeta(board1Id)">
              下载 Meta JSON
            </NButton>
          </NSpace>
        </NSpace>

        <!-- Board 2 -->
        <NSpace vertical :size="8">
          <NButton
            type="primary"
            :loading="generating2"
            @click="handleGenerateBoard(2)"
          >
            生成校准板 2
          </NButton>
          <NSpace v-if="board2Id" :size="12">
            <NButton type="success" size="small" @click="download3mf(board2Id)">
              下载 3MF 模型
            </NButton>
            <NButton type="info" size="small" @click="downloadMeta(board2Id)">
              下载 Meta JSON
            </NButton>
          </NSpace>
        </NSpace>
      </NSpace>
    </NCard>

    <!-- Section 2: Instructions -->
    <NCard title="使用说明">
      <NSteps :current="0" size="small" style="margin-bottom: 16px">
        <NStep title="生成校准板" description="确认八色名称和材质，分别生成校准板 1 和校准板 2" />
        <NStep title="打印校准板" description="优先打印校准板 1，追求更好效果可继续打印校准板 2" />
        <NStep title="拍摄照片" description="在良好光照条件下拍摄打印好的校准板照片" />
        <NStep title="构建 ColorDB" description="分别上传每张校准板的照片和 Meta 文件，构建对应的 ColorDB" />
        <NStep title="使用" description="在图像转换页面选择构建好的 ColorDB（可多选两个 DB 联合使用）" />
      </NSteps>

      <NAlert type="info" title="提示" style="margin-bottom: 12px">
        每张校准板各自对应一个 Meta 文件。构建 ColorDB 时，请确保照片与 Meta 文件来自同一张校准板。
        在图像转换时，可同时选择两个 ColorDB 以获得最佳色彩覆盖。
      </NAlert>
    </NCard>

    <!-- Section 3: Build ColorDB -->
    <NCard title="构建 ColorDB">
      <NSpace vertical :size="16">
        <NGrid :cols="2" :x-gap="16">
          <NGi>
            <div style="margin-bottom: 8px; font-weight: 500">上传校准板照片</div>
            <NUpload
              accept="image/*"
              :max="1"
              :default-upload="false"
              @change="handleImageUpload"
              list-type="text"
            >
              <NButton>选择图片</NButton>
            </NUpload>
            <div v-if="calibImage" style="color: #18a058; font-size: 12px; margin-top: 4px">
              {{ calibImage.name }}
            </div>
          </NGi>
          <NGi>
            <div style="margin-bottom: 8px; font-weight: 500">上传 Meta JSON 文件</div>
            <NUpload
              accept=".json"
              :max="1"
              :default-upload="false"
              @change="handleMetaUpload"
              list-type="text"
            >
              <NButton>选择 JSON</NButton>
            </NUpload>
            <div v-if="calibMeta" style="color: #18a058; font-size: 12px; margin-top: 4px">
              {{ calibMeta.name }}
            </div>
          </NGi>
        </NGrid>

        <div>
          <div style="margin-bottom: 8px; font-weight: 500">ColorDB 名称</div>
          <NInput
            v-model:value="dbName"
            placeholder="仅限字母、数字和下划线"
            style="max-width: 300px"
          />
        </div>

        <NSpace>
          <NButton
            type="primary"
            :loading="building"
            :disabled="!canBuild"
            @click="handleBuildAndNotify"
          >
            构建 ColorDB
          </NButton>
        </NSpace>

        <NSpin :show="building">
          <NAlert v-if="builtDB" type="success" title="构建成功">
            <p>名称: {{ builtDB.name }}</p>
            <p>通道数: {{ builtDB.num_channels }}</p>
            <p>条目数: {{ builtDB.num_entries }}</p>
            <NSpace style="margin-top: 12px">
              <NButton size="small" type="info" @click="downloadBuiltDB">
                下载 ColorDB JSON
              </NButton>
            </NSpace>
            <p style="color: #999; font-size: 12px; margin-top: 8px">
              已自动添加到当前会话的可用数据库列表中，可在"图像转换"页面使用。
            </p>
          </NAlert>
          <NAlert v-if="buildError" type="error" title="构建失败" style="margin-top: 8px; white-space: pre-wrap">
            {{ buildError }}
          </NAlert>
        </NSpin>
      </NSpace>
    </NCard>

    <!-- Section 4: Upload existing ColorDB -->
    <NCard title="上传 ColorDB">
      <NSpace vertical :size="16">
        <NAlert type="info" :bordered="false">
          如果你已经有一个 ColorDB JSON 文件（之前构建并下载的），可以直接上传使用，无需重新构建。
        </NAlert>
        <NGrid :cols="2" :x-gap="16">
          <NGi>
            <div style="margin-bottom: 8px; font-weight: 500">选择 ColorDB JSON 文件</div>
            <NUpload
              accept=".json"
              :max="1"
              :default-upload="false"
              @change="handleUploadFileChange"
              list-type="text"
            >
              <NButton>选择文件</NButton>
            </NUpload>
            <div v-if="uploadFile" style="color: #18a058; font-size: 12px; margin-top: 4px">
              {{ uploadFile.name }}
            </div>
          </NGi>
          <NGi>
            <div style="margin-bottom: 8px; font-weight: 500">名称（可选，覆盖文件中的名称）</div>
            <NInput
              v-model:value="uploadName"
              placeholder="留空则使用文件中的名称"
              style="max-width: 300px"
            />
          </NGi>
        </NGrid>
        <NSpace>
          <NButton
            type="primary"
            :loading="uploading"
            :disabled="!uploadFile"
            @click="handleUpload"
          >
            上传 ColorDB
          </NButton>
        </NSpace>
      </NSpace>
    </NCard>
  </NSpace>
</template>
