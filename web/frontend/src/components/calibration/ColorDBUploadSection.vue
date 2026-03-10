<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NCheckbox,
  NEmpty,
  NForm,
  NFormItem,
  NList,
  NListItem,
  NPopconfirm,
  NSpace,
  NTag,
  NText,
  NUpload,
  useMessage,
} from 'naive-ui'
import { useI18n } from 'vue-i18n'
import { useColorDBUploadFlow } from '../../composables/useColorDBUploadFlow'
import { fetchSessionColorDBs, deleteSessionColorDB } from '../../api/session'
import { roundTo } from '../../runtime/number'
import type { ColorDBInfo } from '../../types'

const { t } = useI18n()

const props = withDefaults(
  defineProps<{
    title?: string
    tips?: string
  }>(),
  {
    title: '',
    tips: '',
  },
)

const emit = defineEmits<{
  (e: 'colordb-updated'): void
  (e: 'go-convert'): void
}>()

const message = useMessage()
const uploadCompleted = ref(false)

const sessionDbs = ref<ColorDBInfo[]>([])
const loadingDbs = ref(false)
const deletingName = ref<string | null>(null)
const selectedNames = ref<Set<string>>(new Set())
const batchDeleting = ref(false)

const allSelected = computed(
  () => sessionDbs.value.length > 0 && selectedNames.value.size === sessionDbs.value.length,
)
const someSelected = computed(
  () => selectedNames.value.size > 0 && selectedNames.value.size < sessionDbs.value.length,
)

function toggleSelect(name: string, checked: boolean) {
  if (checked) {
    selectedNames.value.add(name)
  } else {
    selectedNames.value.delete(name)
  }
  selectedNames.value = new Set(selectedNames.value)
}

function toggleSelectAll(checked: boolean) {
  if (checked) {
    selectedNames.value = new Set(sessionDbs.value.map((db) => db.name))
  } else {
    selectedNames.value = new Set()
  }
}

async function loadSessionDbs() {
  loadingDbs.value = true
  try {
    sessionDbs.value = await fetchSessionColorDBs()
  } catch {
    sessionDbs.value = []
  } finally {
    loadingDbs.value = false
    selectedNames.value = new Set()
  }
}

async function handleDelete(name: string) {
  deletingName.value = name
  try {
    await deleteSessionColorDB(name)
    message.success(t('colordb.upload.messages.deleted', { name }))
    emit('colordb-updated')
    await loadSessionDbs()
  } catch (err: unknown) {
    message.error(
      t('colordb.upload.messages.deleteFailed', {
        error: err instanceof Error ? err.message : String(err),
      }),
    )
  } finally {
    deletingName.value = null
  }
}

async function handleBatchDelete() {
  const names = [...selectedNames.value]
  if (names.length === 0) return
  batchDeleting.value = true
  let ok = 0
  let fail = 0
  for (const name of names) {
    try {
      await deleteSessionColorDB(name)
      ok++
    } catch {
      fail++
    }
  }
  if (ok > 0) emit('colordb-updated')
  if (fail === 0) {
    message.success(t('colordb.upload.messages.batchDeleted', { count: ok }))
  } else {
    message.warning(t('colordb.upload.messages.batchDeletePartial', { ok, fail }))
  }
  batchDeleting.value = false
  await loadSessionDbs()
}

onMounted(loadSessionDbs)

const {
  fileList,
  uploading,
  uploadError,
  canUpload,
  isBatch,
  batchResults,
  handleUploadFileChange,
  handleUpload,
} = useColorDBUploadFlow({
  onUploaded: () => {
    emit('colordb-updated')
  },
  onBatchDone: (results) => {
    const ok = results.filter((r) => r.success)
    const fail = results.filter((r) => !r.success)
    if (ok.length > 0 && fail.length === 0) {
      if (ok.length === 1) {
        const firstOk = ok[0]
        const displayName = firstOk?.dbName ?? firstOk?.fileName ?? 'ColorDB'
        message.success(t('colordb.upload.messages.uploadSuccess', { name: displayName }))
      } else {
        message.success(t('colordb.upload.messages.batchUploadSuccess', { count: ok.length }))
      }
    } else if (ok.length > 0 && fail.length > 0) {
      message.warning(
        t('colordb.upload.messages.batchUploadPartial', { ok: ok.length, fail: fail.length }),
      )
    }
    loadSessionDbs()
  },
})

async function handleUploadAndNotify() {
  const result = await handleUpload()
  if (!result && uploadError.value) {
    message.error(t('colordb.upload.messages.uploadFailed', { error: uploadError.value }))
    return
  }
  if (result && result.length > 0) {
    uploadCompleted.value = true
  }
}

function handleGoToConvert() {
  emit('go-convert')
}
</script>

<template>
  <NCard :title="props.title || t('colordb.upload.defaultTitle')" class="calibration-card">
    <NSpace vertical :size="16">
      <NAlert type="info" :bordered="false">
        {{ props.tips || t('colordb.upload.defaultTips') }}
      </NAlert>

      <NForm label-placement="top" class="calibration-form-block">
        <NFormItem :label="t('colordb.upload.selectLabel')">
          <NUpload
            v-model:file-list="fileList"
            accept=".json"
            multiple
            :default-upload="false"
            list-type="text"
            @change="handleUploadFileChange"
          >
            <NButton>{{ t('colordb.upload.selectButton') }}</NButton>
          </NUpload>
        </NFormItem>
      </NForm>

      <NAlert v-if="isBatch" type="info" :bordered="false">
        {{ t('colordb.upload.selectedCount', { count: fileList.length }) }}
      </NAlert>

      <div class="calibration-actions">
        <NButton
          type="primary"
          :loading="uploading"
          :disabled="!canUpload"
          @click="handleUploadAndNotify"
        >
          {{
            isBatch
              ? t('colordb.upload.uploadAll', { count: fileList.length })
              : t('colordb.upload.uploadSingle')
          }}
        </NButton>
        <NButton v-if="uploadCompleted" type="success" secondary @click="handleGoToConvert">
          {{ t('colordb.upload.goConvert') }}
        </NButton>
      </div>

      <NAlert
        v-if="uploadError"
        type="error"
        :title="t('colordb.upload.failedTitle')"
        class="calibration-error-alert"
      >
        {{ uploadError }}
      </NAlert>

      <div v-if="batchResults.length > 1" class="batch-results">
        <NList bordered size="small">
          <NListItem v-for="(r, i) in batchResults" :key="i">
            <NSpace align="center" :size="8">
              <NTag :type="r.success ? 'success' : 'error'" size="small" :bordered="false">
                {{ r.success ? t('common.success') : t('common.failure') }}
              </NTag>
              <NText>{{ r.fileName }}</NText>
              <NText v-if="r.success && r.dbName" depth="3" style="font-size: 12px">
                → {{ r.dbName }}
              </NText>
              <NText v-if="!r.success && r.error" type="error" style="font-size: 12px">
                {{ r.error }}
              </NText>
            </NSpace>
          </NListItem>
        </NList>
      </div>
    </NSpace>
  </NCard>

  <NCard :title="t('colordb.upload.listTitle')" class="calibration-card session-db-card">
    <NSpace vertical :size="12">
      <NText depth="3" style="font-size: 13px">
        {{ t('colordb.upload.listHint') }}
      </NText>

      <NEmpty
        v-if="!loadingDbs && sessionDbs.length === 0"
        :description="t('colordb.upload.emptyHint')"
      />

      <template v-else>
        <div v-if="sessionDbs.length > 1" class="session-db-toolbar">
          <NCheckbox
            :checked="allSelected"
            :indeterminate="someSelected"
            @update:checked="toggleSelectAll"
          >
            {{ t('colordb.upload.selectAll') }}
          </NCheckbox>
          <NPopconfirm
            v-if="selectedNames.size > 0"
            :positive-text="t('common.confirm')"
            :negative-text="t('common.cancel')"
            @positive-click="handleBatchDelete"
          >
            <template #trigger>
              <NButton size="small" type="error" secondary :loading="batchDeleting">
                {{ t('colordb.upload.deleteSelected', { count: selectedNames.size }) }}
              </NButton>
            </template>
            {{ t('colordb.upload.deleteConfirm', { count: selectedNames.size }) }}
          </NPopconfirm>
        </div>

        <NList bordered hoverable>
          <NListItem v-for="db in sessionDbs" :key="db.name">
            <template #prefix>
              <NSpace align="center" :size="8">
                <NCheckbox
                  :checked="selectedNames.has(db.name)"
                  @update:checked="(v: boolean) => toggleSelect(db.name, v)"
                />
                <NTag size="small" type="info" :bordered="false"
                  >{{ db.num_entries }} {{ t('colordb.upload.entries') }}</NTag
                >
              </NSpace>
            </template>
            <NSpace vertical :size="2">
              <NText strong>{{ db.name }}</NText>
              <NText depth="3" style="font-size: 12px">
                {{ db.num_channels }} {{ t('colordb.upload.channels') }} · {{ db.max_color_layers }}
                {{ t('colordb.upload.colors') }} · {{ roundTo(db.layer_height_mm, 3) }}mm
                {{ t('colordb.upload.layerHeight') }} · {{ roundTo(db.line_width_mm, 3) }}mm
                {{ t('colordb.upload.lineWidth') }}
              </NText>
            </NSpace>
            <template #suffix>
              <NPopconfirm
                :positive-text="t('common.confirm')"
                :negative-text="t('common.cancel')"
                @positive-click="handleDelete(db.name)"
              >
                <template #trigger>
                  <NButton size="small" type="error" quaternary :loading="deletingName === db.name">
                    {{ t('common.delete') }}
                  </NButton>
                </template>
                {{ t('colordb.upload.deleteItemConfirm', { name: db.name }) }}
              </NPopconfirm>
            </template>
          </NListItem>
        </NList>
      </template>

      <NButton size="small" quaternary :loading="loadingDbs" @click="loadSessionDbs">
        {{ t('colordb.upload.refreshList') }}
      </NButton>
    </NSpace>
  </NCard>
</template>
