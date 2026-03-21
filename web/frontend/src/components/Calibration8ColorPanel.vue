<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  NAlert,
  NButton,
  NCard,
  NDivider,
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
import {
  generate8ColorBoard,
  getBoardMetaPath,
  getBoardModelPath,
} from '../services/calibrationService'
import type { FaceOrientation, NozzleSize, PaletteChannel } from '../types'

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

const DEFAULT_8_COLORS: EditablePaletteChannel[] = [
  createChannel('Bamboo Green', 'PLA Basic'),
  createChannel('Black', 'PLA Basic'),
  createChannel('Blue', 'PLA Basic'),
  createChannel('Cyan', 'PLA Basic'),
  createChannel('Magenta', 'PLA Basic'),
  createChannel('Red', 'PLA Basic'),
  createChannel('White', 'PLA Basic'),
  createChannel('Yellow', 'PLA Basic'),
]

const { t } = useI18n()
const message = useMessage()
const { downloadByUrl } = useBlobDownload((error) => message.error(error))

const emit = defineEmits<{
  (e: 'go-colordb-build'): void
}>()

const nozzleSize = ref<NozzleSize>('n04')
const faceOrientation = ref<FaceOrientation>('faceup')

const palette = ref<EditablePaletteChannel[]>(DEFAULT_8_COLORS.map((item) => ({ ...item })))
const board1Id = ref<string | null>(null)
const board2Id = ref<string | null>(null)
const generating1 = ref(false)
const generating2 = ref(false)
const currentStep = computed(() => {
  if (board1Id.value || board2Id.value) return 2
  return 1
})

const nextActionHint = computed(() => {
  if (!board1Id.value) return t('calibration.eightColor.nextHint.step1')
  return t('calibration.eightColor.nextHint.step2')
})

function toPaletteRequest(paletteRows: EditablePaletteChannel[]): PaletteChannel[] {
  return paletteRows.map((row) => ({
    color: row.color,
    material: row.material,
  }))
}

function validatePalette(): boolean {
  for (const channel of palette.value) {
    if (!channel.color.trim()) {
      message.error(t('calibration.fillAllColors'))
      return false
    }
  }
  return true
}

async function handleGenerateBoard(boardIndex: number) {
  if (!validatePalette()) return
  const isBoard1 = boardIndex === 1
  const generatingRef = isBoard1 ? generating1 : generating2
  const boardRef = isBoard1 ? board1Id : board2Id

  generatingRef.value = true
  boardRef.value = null
  try {
    const response = await generate8ColorBoard({
      palette: toPaletteRequest(palette.value),
      board_index: boardIndex,
      nozzle_size: nozzleSize.value,
      face_orientation: faceOrientation.value,
    })
    boardRef.value = response.board_id
    message.success(t('calibration.eightColor.boardGenerateSuccess', { index: boardIndex }))
  } catch (error: unknown) {
    message.error(
      t('calibration.generateFailed', {
        error: error instanceof Error ? error.message : String(error),
      }),
    )
  } finally {
    generatingRef.value = false
  }
}

async function download3mf(boardId: string | null) {
  if (!boardId) return
  const ch = palette.value.length
  await downloadByUrl(
    getBoardModelPath(boardId),
    `calibration-board-${ch}ch-${boardId.slice(0, 8)}.3mf`,
  )
}

async function downloadMeta(boardId: string | null) {
  if (!boardId) return
  const ch = palette.value.length
  await downloadByUrl(
    getBoardMetaPath(boardId),
    `calibration-board-${ch}ch-${boardId.slice(0, 8)}-meta.json`,
  )
}

function handleResetCalibrationParams() {
  nozzleSize.value = 'n04'
  faceOrientation.value = 'faceup'
  palette.value = DEFAULT_8_COLORS.map((item) => ({ ...item }))
}
</script>

<template>
  <NSpace vertical :size="20" class="calibration-layout">
    <NCard class="calibration-card">
      <template #header>
        <div class="calibration-header">
          <NSpace align="center" :size="8">
            <span>{{ t('calibration.eightColor.title') }}</span>
          </NSpace>
          <p class="calibration-subtitle">
            {{ t('calibration.currentStep', { step: currentStep }) }}
          </p>
        </div>
      </template>

      <NSteps :current="currentStep" size="small" class="calibration-steps">
        <NStep
          :title="t('calibration.eightColor.steps.generate.title')"
          :description="t('calibration.eightColor.steps.generate.description')"
        />
        <NStep
          :title="t('calibration.eightColor.steps.print.title')"
          :description="t('calibration.eightColor.steps.print.description')"
        />
        <NStep
          :title="t('calibration.eightColor.steps.build.title')"
          :description="t('calibration.eightColor.steps.build.description')"
        />
      </NSteps>

      <NAlert type="info" :bordered="false" class="calibration-step-alert">
        {{ nextActionHint }}
      </NAlert>
    </NCard>

    <NCard :title="t('calibration.eightColor.step1Title')" class="calibration-card">
      <template #header-extra>
        <NButton
          size="tiny"
          quaternary
          :disabled="generating1 || generating2"
          @click="handleResetCalibrationParams"
        >
          {{ t('common.reset') }}
        </NButton>
      </template>
      <NSpace vertical :size="12">
        <NAlert type="info" :bordered="false">
          {{ t('calibration.eightColor.tip') }}
        </NAlert>

        <div class="calibration-preset-row">
          <span class="calibration-preset-label">{{ t('calibration.nozzleSize') }}</span>
          <NRadioGroup v-model:value="nozzleSize" size="small">
            <NRadioButton value="n04">0.4mm</NRadioButton>
            <NRadioButton value="n02">0.2mm</NRadioButton>
          </NRadioGroup>
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
        </div>

        <NDivider />

        <div class="calibration-actions">
          <NButton type="primary" :loading="generating1" @click="handleGenerateBoard(1)">
            {{ t('calibration.eightColor.generateBoard1') }}
          </NButton>
          <NTag v-if="board1Id" type="success" :bordered="false" size="small">{{
            t('calibration.eightColor.board1Generated')
          }}</NTag>
        </div>

        <div v-if="board1Id" class="calibration-actions">
          <NButton type="success" size="small" @click="download3mf(board1Id)">{{
            t('calibration.eightColor.downloadBoard1_3mf')
          }}</NButton>
          <NButton type="info" size="small" @click="downloadMeta(board1Id)">{{
            t('calibration.eightColor.downloadBoard1_meta')
          }}</NButton>
        </div>

        <div class="calibration-actions">
          <NButton type="primary" :loading="generating2" @click="handleGenerateBoard(2)">
            {{ t('calibration.eightColor.generateBoard2') }}
          </NButton>
          <NTag v-if="board2Id" type="success" :bordered="false" size="small">{{
            t('calibration.eightColor.board2Generated')
          }}</NTag>
        </div>

        <div v-if="board2Id" class="calibration-actions">
          <NButton type="success" size="small" @click="download3mf(board2Id)">{{
            t('calibration.eightColor.downloadBoard2_3mf')
          }}</NButton>
          <NButton type="info" size="small" @click="downloadMeta(board2Id)">{{
            t('calibration.eightColor.downloadBoard2_meta')
          }}</NButton>
        </div>
      </NSpace>
    </NCard>

    <NCard :title="t('calibration.eightColor.step2Title')" class="calibration-card">
      <NSpace vertical :size="12">
        <NAlert type="warning" :bordered="false" :title="t('calibration.eightColor.keyPoints')">
          {{ t('calibration.eightColor.keyPointText1') }}
        </NAlert>
        <NAlert type="info" :bordered="false">
          {{ t('calibration.eightColor.keyPointText2') }}
        </NAlert>
      </NSpace>
    </NCard>

    <NCard :title="t('calibration.eightColor.step3Title')" class="calibration-card">
      <NSpace vertical :size="12">
        <NAlert type="info" :bordered="false">
          {{ t('calibration.eightColor.step3GoHint') }}
        </NAlert>
        <div class="calibration-actions">
          <NButton type="primary" @click="emit('go-colordb-build')">
            {{ t('calibration.goToColorDBBuild') }}
          </NButton>
        </div>
      </NSpace>
    </NCard>
  </NSpace>
</template>
