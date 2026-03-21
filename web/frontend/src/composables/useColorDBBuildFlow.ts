import { computed, ref, shallowRef } from 'vue'
import { useI18n } from 'vue-i18n'
import type { UploadFileInfo } from 'naive-ui'
import { buildColorDB, locateCalibrationBoard } from '../api/calibration'
import type { ColorDBBuildResult, CalibrationLocateResult } from '../types'
import { isValidColorDBName } from './colordbName'

export type BuildPhase = 'upload' | 'locating' | 'preview' | 'building' | 'result'

export interface BoardGeometryInfo {
  gridRows: number
  gridCols: number
  boardW: number
  boardH: number
  margin: number
  tile: number
  gap: number
  offset: number
}

export interface MetaSummaryInfo {
  name: string
  gridRows: number
  gridCols: number
  numChannels: number
  colorLayers: number
}

type UseColorDBBuildFlowOptions = {
  onBuilt?: (result: ColorDBBuildResult) => void | Promise<void>
}

function parseMetaGeometry(metaJson: string): BoardGeometryInfo {
  const meta = JSON.parse(metaJson)
  const scale = meta.config?.layout?.resolution_scale ?? 8
  const tileFactor = meta.config?.layout?.tile_factor ?? 10
  const gapFactor = meta.config?.layout?.gap_factor ?? 2
  const marginFactor = meta.config?.layout?.margin_factor ?? 24
  const offsetFactor = meta.config?.layout?.fiducial?.offset_factor ?? 14

  const gridRows = meta.grid_rows ?? 0
  const gridCols = meta.grid_cols ?? 0

  const tile = tileFactor * scale
  const gap = gapFactor * scale
  const margin = marginFactor * scale
  const offset = offsetFactor * scale

  const colorW = gridCols * tile + (gridCols - 1) * gap
  const colorH = gridRows * tile + (gridRows - 1) * gap
  const boardW = colorW + 2 * margin
  const boardH = colorH + 2 * margin

  return { gridRows, gridCols, boardW, boardH, margin, tile, gap, offset }
}

function parseMetaSummary(metaJson: string): MetaSummaryInfo {
  const meta = JSON.parse(metaJson)
  return {
    name: meta.name ?? '',
    gridRows: meta.grid_rows ?? 0,
    gridCols: meta.grid_cols ?? 0,
    numChannels: meta.config?.recipe?.num_channels ?? 0,
    colorLayers: meta.config?.recipe?.color_layers ?? 0,
  }
}

export function useColorDBBuildFlow(options: UseColorDBBuildFlowOptions = {}) {
  const { t } = useI18n()
  const calibImage = ref<File | null>(null)
  const calibMeta = ref<File | null>(null)
  const dbName = ref('')
  const phase = ref<BuildPhase>('upload')
  const building = ref(false)
  const builtDB = shallowRef<ColorDBBuildResult | null>(null)
  const buildError = ref<string | null>(null)
  const locateError = ref<string | null>(null)
  const corners = ref<[number, number][]>([])
  const locateResult = shallowRef<CalibrationLocateResult | null>(null)
  const boardGeometry = shallowRef<BoardGeometryInfo | null>(null)
  const metaSummary = shallowRef<MetaSummaryInfo | null>(null)
  const metaParseError = ref<string | null>(null)
  const metaJsonCache = ref<string | null>(null)
  const uploadResetKey = ref(0)

  const dbNameError = computed(() => {
    const value = dbName.value.trim()
    if (!value) return t('colordb.build.nameRequired')
    if (!isValidColorDBName(value)) return t('colordb.build.nameInvalid')
    return ''
  })

  const canLocate = computed(() =>
    Boolean(calibImage.value && calibMeta.value && boardGeometry.value),
  )

  const canBuild = computed(
    () => phase.value === 'preview' && corners.value.length === 4 && dbNameError.value.length === 0,
  )

  function resetLocateState() {
    corners.value = []
    locateResult.value = null
    locateError.value = null
    builtDB.value = null
    buildError.value = null
    phase.value = 'upload'
  }

  function handleImageUpload({ file }: { file: UploadFileInfo }) {
    calibImage.value = file.status === 'removed' ? null : (file.file ?? null)
    resetLocateState()
    return false
  }

  function handleMetaUpload({ file }: { file: UploadFileInfo }) {
    const newFile = file.status === 'removed' ? null : (file.file ?? null)
    calibMeta.value = newFile
    resetLocateState()

    if (newFile) {
      newFile
        .text()
        .then((text) => {
          metaJsonCache.value = text
          try {
            boardGeometry.value = parseMetaGeometry(text)
            metaSummary.value = parseMetaSummary(text)
            metaParseError.value = null
          } catch (e: unknown) {
            boardGeometry.value = null
            metaSummary.value = null
            metaParseError.value = e instanceof Error ? e.message : String(e)
          }
        })
        .catch(() => {
          boardGeometry.value = null
          metaSummary.value = null
          metaJsonCache.value = null
          metaParseError.value = t('colordb.meta.readFailed')
        })
    } else {
      boardGeometry.value = null
      metaSummary.value = null
      metaJsonCache.value = null
      metaParseError.value = null
    }

    return false
  }

  async function handleLocate(): Promise<boolean> {
    if (!canLocate.value || !calibImage.value || !calibMeta.value) return false

    phase.value = 'locating'
    locateError.value = null
    buildError.value = null

    try {
      if (!boardGeometry.value && metaJsonCache.value) {
        boardGeometry.value = parseMetaGeometry(metaJsonCache.value)
      }

      const result = await locateCalibrationBoard(calibImage.value, calibMeta.value)
      locateResult.value = result
      corners.value = result.corners.map((c) => [c[0], c[1]] as [number, number])
      phase.value = 'preview'
      return true
    } catch (error: unknown) {
      locateError.value = error instanceof Error ? error.message : String(error)
      phase.value = 'upload'
      return false
    }
  }

  async function handleBuild(): Promise<ColorDBBuildResult | null> {
    if (!canBuild.value || !calibImage.value || !calibMeta.value) return null

    phase.value = 'building'
    building.value = true
    builtDB.value = null
    buildError.value = null
    try {
      const result = await buildColorDB(
        calibImage.value,
        calibMeta.value,
        dbName.value.trim(),
        corners.value,
      )
      builtDB.value = result
      phase.value = 'result'
      if (options.onBuilt) {
        await options.onBuilt(result)
      }
      return result
    } catch (error: unknown) {
      buildError.value = error instanceof Error ? error.message : String(error)
      phase.value = 'preview'
      return null
    } finally {
      building.value = false
    }
  }

  function updateCorners(newCorners: [number, number][]) {
    corners.value = newCorners
  }

  function backToUpload() {
    calibImage.value = null
    calibMeta.value = null
    dbName.value = ''
    phase.value = 'upload'
    builtDB.value = null
    buildError.value = null
    locateError.value = null
    corners.value = []
    locateResult.value = null
    boardGeometry.value = null
    metaSummary.value = null
    metaParseError.value = null
    metaJsonCache.value = null
    uploadResetKey.value++
  }

  return {
    calibImage,
    calibMeta,
    dbName,
    dbNameError,
    phase,
    building,
    builtDB,
    buildError,
    locateError,
    corners,
    locateResult,
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
  }
}
