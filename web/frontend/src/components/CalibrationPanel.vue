<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NDivider,
  NImage,
  NInput,
  NRadioButton,
  NRadioGroup,
  NSpace,
  NStep,
  NSteps,
  NTag,
  useMessage,
} from 'naive-ui'
import { useI18n } from 'vue-i18n'
import { useBlobDownload } from '../composables/useBlobDownload'
import { generateBoard, getBoardMetaPath, getBoardModelPath } from '../services/calibrationService'
import type { FaceOrientation, NozzleSize, PaletteChannel } from '../types'
import ColorDBBuildSection from './calibration/ColorDBBuildSection.vue'

type EditablePaletteChannel = PaletteChannel & {
  id: number
}

let nextPaletteId = 0

function createChannel(color: string, material: string): EditablePaletteChannel {
  nextPaletteId += 1
  return {
    id: nextPaletteId,
    color,
    material,
  }
}

const { t } = useI18n()
const message = useMessage()
const { downloadByUrl } = useBlobDownload((error) => message.error(error))

const emit = defineEmits<{
  (e: 'colordb-updated'): void
}>()

const nozzleSize = ref<NozzleSize>('n04')
const faceOrientation = ref<FaceOrientation>('faceup')

const maxChannels = 4
const minChannels = 2
const hasReadyColorDB = ref(false)
const boardId = ref<string | null>(null)
const generating = ref(false)

const palette = ref<EditablePaletteChannel[]>([
  createChannel('White', 'PLA Basic'),
  createChannel('Yellow', 'PLA Basic'),
  createChannel('Red', 'PLA Basic'),
  createChannel('Blue', 'PLA Basic'),
])

const currentStep = computed(() => {
  if (hasReadyColorDB.value) return 3
  if (boardId.value) return 2
  return 1
})

const nextActionHint = computed(() => {
  if (!boardId.value) return t('calibration.fourColor.nextHint.step1')
  if (!hasReadyColorDB.value) return t('calibration.fourColor.nextHint.step2')
  return t('calibration.fourColor.nextHint.step3')
})

function toPaletteRequest(paletteRows: EditablePaletteChannel[]): PaletteChannel[] {
  return paletteRows.map((row) => ({
    color: row.color,
    material: row.material,
  }))
}

function addChannel() {
  if (palette.value.length >= maxChannels) {
    message.warning(t('calibration.fourColor.maxChannels', { max: maxChannels }))
    return
  }
  palette.value.push(createChannel('', 'PLA Basic'))
}

function removeChannel(index: number) {
  if (palette.value.length <= minChannels) {
    message.warning(t('calibration.fourColor.minChannels', { min: minChannels }))
    return
  }
  palette.value.splice(index, 1)
}

async function handleGenerate() {
  for (const channel of palette.value) {
    if (!channel.color.trim()) {
      message.error(t('calibration.fillAllColors'))
      return
    }
  }

  generating.value = true
  boardId.value = null
  try {
    const response = await generateBoard({
      palette: toPaletteRequest(palette.value),
      nozzle_size: nozzleSize.value,
      face_orientation: faceOrientation.value,
    })
    boardId.value = response.board_id
    message.success(t('calibration.generateSuccess'))
  } catch (error: unknown) {
    message.error(
      t('calibration.generateFailed', {
        error: error instanceof Error ? error.message : String(error),
      }),
    )
  } finally {
    generating.value = false
  }
}

async function download3mf() {
  if (!boardId.value) return
  await downloadByUrl(
    getBoardModelPath(boardId.value),
    `calibration-board-${boardId.value.slice(0, 8)}.3mf`,
  )
}

async function downloadMeta() {
  if (!boardId.value) return
  await downloadByUrl(
    getBoardMetaPath(boardId.value),
    `calibration-board-${boardId.value.slice(0, 8)}.json`,
  )
}

function handleResetCalibrationParams() {
  nozzleSize.value = 'n04'
  faceOrientation.value = 'faceup'
  palette.value = [
    createChannel('White', 'PLA Basic'),
    createChannel('Yellow', 'PLA Basic'),
    createChannel('Red', 'PLA Basic'),
    createChannel('Blue', 'PLA Basic'),
  ]
}

function handleColorDBUpdated() {
  hasReadyColorDB.value = true
  emit('colordb-updated')
}
</script>

<template>
  <NSpace vertical :size="20" class="calibration-layout">
    <NCard class="calibration-card">
      <template #header>
        <div class="calibration-header">
          <NSpace align="center" :size="8">
            <span>{{ t('calibration.fourColor.title') }}</span>
            <NTag size="small" :bordered="false" type="info">{{
              t('calibration.fourColor.tag')
            }}</NTag>
          </NSpace>
          <p class="calibration-subtitle">
            {{ t('calibration.currentStep', { step: currentStep }) }}
          </p>
        </div>
      </template>

      <NSteps :current="currentStep" size="small" class="calibration-steps">
        <NStep
          :title="t('calibration.fourColor.steps.generate.title')"
          :description="t('calibration.fourColor.steps.generate.description')"
        />
        <NStep
          :title="t('calibration.fourColor.steps.print.title')"
          :description="t('calibration.fourColor.steps.print.description')"
        />
        <NStep
          :title="t('calibration.fourColor.steps.build.title')"
          :description="t('calibration.fourColor.steps.build.description')"
        />
      </NSteps>

      <NAlert type="info" :bordered="false" class="calibration-step-alert">
        {{ nextActionHint }}
      </NAlert>
    </NCard>

    <NCard :title="t('calibration.fourColor.step1Title')" class="calibration-card">
      <template #header-extra>
        <NButton
          size="tiny"
          quaternary
          :disabled="generating"
          @click="handleResetCalibrationParams"
          >{{ t('common.reset') }}</NButton
        >
      </template>
      <NSpace vertical :size="12">
        <div class="calibration-preset-row">
          <span class="calibration-preset-label">{{ t('calibration.nozzleSize') }}</span>
          <NRadioGroup v-model:value="nozzleSize" size="small">
            <NRadioButton value="n04">0.4mm</NRadioButton>
            <NRadioButton value="n02">0.2mm</NRadioButton>
          </NRadioGroup>
        </div>

        <div class="calibration-preset-row">
          <span class="calibration-preset-label">{{ t('calibration.faceDirection') }}</span>
          <NRadioGroup v-model:value="faceOrientation" size="small">
            <NRadioButton value="faceup">{{ t('calibration.faceUp') }}</NRadioButton>
            <NRadioButton value="facedown">{{ t('calibration.faceDown') }}</NRadioButton>
          </NRadioGroup>
        </div>

        <NDivider style="margin: 4px 0" />

        <div v-for="(channel, index) in palette" :key="channel.id" class="calibration-palette-row">
          <NTag :bordered="false" type="info" size="small">{{ index + 1 }}</NTag>
          <NInput
            v-model:value="channel.color"
            :placeholder="t('calibration.colorPlaceholder')"
            class="calibration-palette-input"
          />
          <NInput
            v-model:value="channel.material"
            :placeholder="t('calibration.materialPlaceholder')"
            class="calibration-palette-input"
          />
          <NButton size="small" quaternary type="error" @click="removeChannel(index)">{{
            t('common.delete')
          }}</NButton>
        </div>

        <div class="calibration-inline-actions">
          <NButton size="small" :disabled="palette.length >= maxChannels" @click="addChannel">
            {{ t('calibration.fourColor.addChannel') }}
          </NButton>
        </div>

        <NDivider />

        <div class="calibration-actions">
          <NButton type="primary" :loading="generating" @click="handleGenerate">
            {{ t('calibration.generateBoard') }}
          </NButton>
          <NTag v-if="boardId" type="success" :bordered="false" size="small">{{
            t('calibration.boardGenerated')
          }}</NTag>
        </div>

        <div v-if="boardId" class="calibration-actions">
          <NButton type="success" @click="download3mf">{{ t('calibration.download3mf') }}</NButton>
          <NButton type="info" @click="downloadMeta">{{ t('calibration.downloadMeta') }}</NButton>
        </div>
      </NSpace>
    </NCard>

    <NCard :title="t('calibration.fourColor.step2Title')" class="calibration-card">
      <NSpace vertical :size="14">
        <NAlert type="warning" :title="t('calibration.fourColor.notice')" :bordered="false">
          {{ t('calibration.fourColor.noticeText') }}
        </NAlert>

        <NDivider>{{ t('calibration.fourColor.exampleTitle') }}</NDivider>
        <div class="calibration-image-wrap">
          <NImage
            src="/examples/RYBW_new.jpg"
            object-fit="contain"
            :img-props="{
              style: 'max-width: 100%; max-height: 380px; object-fit: contain; cursor: zoom-in;',
            }"
            :alt="t('calibration.fourColor.exampleAlt')"
          />
          <p class="calibration-image-caption">
            {{ t('calibration.fourColor.exampleDesc') }}
          </p>
        </div>
      </NSpace>
    </NCard>

    <ColorDBBuildSection
      :title="t('calibration.fourColor.step3Title')"
      :tips="t('calibration.fourColor.step3Tips')"
      :build-button-text="t('calibration.fourColor.step3BuildButton')"
      @colordb-updated="handleColorDBUpdated"
    />
  </NSpace>
</template>
