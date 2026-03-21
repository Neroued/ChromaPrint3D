<script setup lang="ts">
import {
  NAlert,
  NButton,
  NCard,
  NForm,
  NFormItem,
  NGrid,
  NGi,
  NInput,
  NSpace,
  NTag,
  NUpload,
  useMessage,
} from 'naive-ui'
import { useI18n } from 'vue-i18n'
import { useBlobDownload } from '../../composables/useBlobDownload'
import { useColorDBBuildFlow } from '../../composables/useColorDBBuildFlow'
import { getSessionColorDBDownloadPath } from '../../services/sessionColorDBService'
import CalibrationLocatePreview from './CalibrationLocatePreview.vue'
import ColorDBResultOverview from './ColorDBResultOverview.vue'

const { t } = useI18n()

const props = withDefaults(
  defineProps<{
    title?: string
    tips?: string
    buildButtonText?: string
  }>(),
  {
    title: '',
    tips: '',
    buildButtonText: '',
  },
)

const emit = defineEmits<{
  (e: 'colordb-updated'): void
}>()

const message = useMessage()
const { downloadByUrl } = useBlobDownload((error) => message.error(error))

const {
  calibImage,
  dbName,
  dbNameError,
  phase,
  building,
  builtDB,
  buildError,
  locateError,
  corners,
  boardGeometry,
  metaSummary,
  metaParseError,
  uploadResetKey,
  canLocate,
  canBuild,
  handleImageUpload,
  handleMetaUpload,
  handleLocate,
  handleBuild,
  updateCorners,
  backToUpload,
} = useColorDBBuildFlow({
  onBuilt: (result) => {
    message.success(t('colordb.build.messages.success', { name: result.name }))
    emit('colordb-updated')
  },
})

async function handleLocateAndNotify() {
  const ok = await handleLocate()
  if (!ok && locateError.value) {
    message.error(t('colordb.locate.failed', { error: locateError.value }))
  }
}

async function handleBuildAndNotify() {
  const result = await handleBuild()
  if (!result && buildError.value) {
    message.error(t('colordb.build.messages.failed', { error: buildError.value }))
  }
}

async function downloadBuiltDB() {
  if (!builtDB.value) return
  await downloadByUrl(
    getSessionColorDBDownloadPath(builtDB.value.name),
    `${builtDB.value.name}.json`,
  )
}
</script>

<template>
  <NCard :title="props.title || t('colordb.build.defaultTitle')" class="calibration-card">
    <NSpace vertical :size="16">
      <NAlert v-if="props.tips" type="info" :bordered="false">
        {{ props.tips }}
      </NAlert>

      <!-- Upload + Name (hidden in result phase) -->
      <template v-if="phase !== 'result'">
        <NGrid :cols="2" :x-gap="16" :y-gap="12" responsive="screen" item-responsive>
          <NGi span="2 m:1">
            <NForm label-placement="top" class="calibration-form-block">
              <NFormItem :label="t('colordb.build.uploadPhoto')">
                <NUpload
                  :key="`img-${uploadResetKey}`"
                  accept="image/*"
                  :max="1"
                  :default-upload="false"
                  list-type="text"
                  @change="handleImageUpload"
                >
                  <NButton>{{ t('colordb.build.selectImage') }}</NButton>
                </NUpload>
              </NFormItem>
            </NForm>
          </NGi>

          <NGi span="2 m:1">
            <NForm label-placement="top" class="calibration-form-block">
              <NFormItem :label="t('colordb.build.uploadMeta')">
                <NUpload
                  :key="`meta-${uploadResetKey}`"
                  accept=".json"
                  :max="1"
                  :default-upload="false"
                  list-type="text"
                  @change="handleMetaUpload"
                >
                  <NButton>{{ t('colordb.build.selectJson') }}</NButton>
                </NUpload>
              </NFormItem>
            </NForm>
          </NGi>
        </NGrid>

        <!-- Meta summary -->
        <div v-if="metaSummary" class="calibration-meta-summary">
          <NSpace :size="8" align="center">
            <NTag :bordered="false" size="small" type="info">
              {{ t('colordb.meta.grid') }}: {{ metaSummary.gridRows }}×{{ metaSummary.gridCols }}
            </NTag>
            <NTag :bordered="false" size="small" type="info">
              {{ t('colordb.meta.channels') }}: {{ metaSummary.numChannels }}
            </NTag>
            <NTag :bordered="false" size="small" type="info">
              {{ t('colordb.meta.colorLayers') }}: {{ metaSummary.colorLayers }}
            </NTag>
          </NSpace>
        </div>

        <!-- Meta parse error -->
        <NAlert v-if="metaParseError" type="error" :bordered="false">
          {{ metaParseError }}
        </NAlert>

        <!-- Name input -->
        <NForm label-placement="top" class="calibration-form-block">
          <NFormItem
            :label="t('colordb.build.nameLabel')"
            :validation-status="dbNameError ? 'error' : undefined"
            :feedback="dbNameError || t('colordb.build.nameInvalid')"
          >
            <NInput
              v-model:value="dbName"
              :placeholder="t('colordb.build.namePlaceholder')"
              class="calibration-db-input"
            />
          </NFormItem>
        </NForm>
      </template>

      <!-- Image preview (visible when image is uploaded, hidden in result phase) -->
      <CalibrationLocatePreview
        v-if="calibImage && phase !== 'result'"
        :image-file="calibImage"
        :corners="corners"
        :board-geometry="boardGeometry"
        @update:corners="updateCorners"
      />

      <!-- Locate success alert -->
      <NAlert
        v-if="(phase === 'preview' || phase === 'building') && !locateError"
        type="success"
        :bordered="false"
      >
        {{ t('colordb.locate.success') }}
      </NAlert>

      <!-- Locate error -->
      <NAlert v-if="locateError" type="error" :title="t('colordb.locate.failedTitle')">
        {{ locateError }}
      </NAlert>

      <!-- Action buttons (not in result phase) -->
      <div v-if="phase !== 'result'" class="calibration-actions">
        <NButton
          v-if="phase === 'upload' || phase === 'locating'"
          type="primary"
          :loading="phase === 'locating'"
          :disabled="!canLocate"
          @click="handleLocateAndNotify"
        >
          {{ t('colordb.locate.locateButton') }}
        </NButton>
        <NButton
          v-if="phase === 'preview' || phase === 'building'"
          type="primary"
          :loading="building"
          :disabled="!canBuild"
          @click="handleBuildAndNotify"
        >
          {{ props.buildButtonText || t('colordb.build.defaultButtonText') }}
        </NButton>
      </div>

      <!-- Build error -->
      <NAlert
        v-if="buildError"
        type="error"
        :title="t('colordb.build.failed')"
        class="calibration-error-alert"
      >
        {{ buildError }}
      </NAlert>

      <!-- Result phase -->
      <template v-if="phase === 'result' && builtDB">
        <NAlert type="success" :title="t('colordb.build.success')">
          <p class="calibration-result-line">
            {{ t('colordb.build.successName', { name: builtDB.name }) }}
          </p>
          <p class="calibration-result-line">
            {{ t('colordb.build.successChannels', { channels: builtDB.num_channels }) }}
          </p>
          <p class="calibration-result-line">
            {{ t('colordb.build.successEntries', { entries: builtDB.num_entries }) }}
          </p>
          <div class="calibration-actions calibration-actions--compact">
            <NButton size="small" type="info" @click="downloadBuiltDB">
              {{ t('colordb.build.downloadJson') }}
            </NButton>
            <NButton size="small" @click="backToUpload">
              {{ t('colordb.build.buildAnother') }}
            </NButton>
          </div>
          <p class="calibration-result-tip">
            {{ t('colordb.build.autoAdded') }}
          </p>
        </NAlert>

        <ColorDBResultOverview :result="builtDB" :palette="builtDB.palette" />
      </template>
    </NSpace>
  </NCard>
</template>
