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
  NSpin,
  NSpace,
  NText,
  NUpload,
  useMessage,
} from 'naive-ui'
import { useI18n } from 'vue-i18n'
import { useBlobDownload } from '../../composables/useBlobDownload'
import { useColorDBBuildFlow } from '../../composables/useColorDBBuildFlow'
import { getSessionColorDBDownloadPath } from '../../services/sessionColorDBService'

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
  calibMeta,
  dbName,
  dbNameError,
  building,
  builtDB,
  buildError,
  canBuild,
  handleImageUpload,
  handleMetaUpload,
  handleBuild,
} = useColorDBBuildFlow({
  onBuilt: (result) => {
    message.success(t('colordb.build.messages.success', { name: result.name }))
    emit('colordb-updated')
  },
})

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

      <NGrid :cols="2" :x-gap="16" :y-gap="12" responsive="screen" item-responsive>
        <NGi span="2 m:1">
          <NForm label-placement="top" class="calibration-form-block">
            <NFormItem :label="t('colordb.build.uploadPhoto')">
              <NUpload
                accept="image/*"
                :max="1"
                :default-upload="false"
                list-type="text"
                @change="handleImageUpload"
              >
                <NButton>{{ t('colordb.build.selectImage') }}</NButton>
              </NUpload>
            </NFormItem>
            <NText v-if="calibImage" depth="3" class="calibration-file-name">
              {{ t('colordb.build.selectedFile', { name: calibImage.name }) }}
            </NText>
          </NForm>
        </NGi>

        <NGi span="2 m:1">
          <NForm label-placement="top" class="calibration-form-block">
            <NFormItem :label="t('colordb.build.uploadMeta')">
              <NUpload
                accept=".json"
                :max="1"
                :default-upload="false"
                list-type="text"
                @change="handleMetaUpload"
              >
                <NButton>{{ t('colordb.build.selectJson') }}</NButton>
              </NUpload>
            </NFormItem>
            <NText v-if="calibMeta" depth="3" class="calibration-file-name">
              {{ t('colordb.build.selectedFile', { name: calibMeta.name }) }}
            </NText>
          </NForm>
        </NGi>
      </NGrid>

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

      <div class="calibration-actions">
        <NButton
          type="primary"
          :loading="building"
          :disabled="!canBuild"
          @click="handleBuildAndNotify"
        >
          {{ props.buildButtonText || t('colordb.build.defaultButtonText') }}
        </NButton>
      </div>

      <NSpin :show="building">
        <NAlert v-if="builtDB" type="success" :title="t('colordb.build.success')">
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
            <NButton size="small" type="info" @click="downloadBuiltDB">{{
              t('colordb.build.downloadJson')
            }}</NButton>
          </div>
          <p class="calibration-result-tip">
            {{ t('colordb.build.autoAdded') }}
          </p>
        </NAlert>

        <NAlert
          v-if="buildError"
          type="error"
          :title="t('colordb.build.failed')"
          class="calibration-error-alert"
        >
          {{ buildError }}
        </NAlert>
      </NSpin>
    </NSpace>
  </NCard>
</template>
