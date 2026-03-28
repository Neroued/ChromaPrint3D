import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { storeToRefs } from 'pinia'
import type { SelectOption } from 'naive-ui'
import { createInitialConvertParams } from '../domain/params/convertDefaults'
import { formatFloat, roundTo } from '../runtime/number'
import { useAppStore } from '../stores/app'
import {
  getPixelSizePresets,
  type ColorDBInfo,
  type ConvertAnyParams,
  type DefaultConfig,
  type PaletteChannel,
} from '../types'
import { fetchAvailableColorDBs, fetchParamPanelBootstrap } from '../services/paramService'

interface ChannelOption {
  key: string
  color: string
  material: string
}

interface ChannelPreset {
  label: string
  value: string
  colors: string[] | null
}

const colorSpaceOptions: SelectOption[] = [
  { label: 'Lab', value: 'lab' },
  { label: 'RGB', value: 'rgb' },
]

function channelKey(ch: PaletteChannel): string {
  return `${ch.color}|${ch.material}`
}

export function useParamPanelState() {
  const { t } = useI18n()

  const channelPresets: ChannelPreset[] = [
    { label: t('common.all'), value: 'all', colors: null },
    { label: 'RYBW', value: 'rybw', colors: ['Red', 'Yellow', 'Blue', 'White'] },
    { label: 'CMYW', value: 'cmyw', colors: ['Cyan', 'Magenta', 'Yellow', 'White'] },
  ]

  const ditherOpts: SelectOption[] = [
    { label: t('param.off'), value: 'none' },
    { label: 'Floyd-Steinberg', value: 'floyd_steinberg' },
  ]

  const clusterModeOpts: SelectOption[] = [
    { label: t('param.clusterModeOff'), value: 'off' },
    { label: t('param.clusterModeAuto'), value: 'auto' },
    { label: t('param.clusterModeManual'), value: 'manual' },
  ]

  const translatedTooltips = {
    print_mode: t('param.tooltips.print_mode'),
    color_space: t('param.tooltips.color_space'),
    max_width: t('param.tooltips.max_width'),
    max_height: t('param.tooltips.max_height'),
    target_width_mm: t('param.tooltips.target_width_mm'),
    target_height_mm: t('param.tooltips.target_height_mm'),
    pixel_mm_simple: t('param.tooltips.pixel_mm_simple'),
    cluster_count: t('param.tooltips.cluster_count'),
    dither: t('param.tooltips.dither'),
    dither_strength: t('param.tooltips.dither_strength'),
    db_names: t('param.tooltips.db_names'),
    allowed_channels: t('param.tooltips.allowed_channels'),
    scale: t('param.tooltips.scale'),
    k_candidates: t('param.tooltips.k_candidates'),
    flip_y: t('param.tooltips.flip_y'),
    pixel_mm: t('param.tooltips.pixel_mm'),
    layer_height_mm: t('param.tooltips.layer_height_mm'),
    base_layers: t('param.tooltips.base_layers'),
    double_sided: t('param.tooltips.double_sided'),
    transparent_layer_mm: t('param.tooltips.transparent_layer_mm'),
    model_enable: t('param.tooltips.model_enable'),
    model_only: t('param.tooltips.model_only'),
    model_threshold: t('param.tooltips.model_threshold'),
    model_margin: t('param.tooltips.model_margin'),
    generate_preview: t('param.tooltips.generate_preview'),
    generate_source_mask: t('param.tooltips.generate_source_mask'),
    tessellation_tolerance_mm: t('param.tooltips.tessellation_tolerance_mm'),
  }

  function buildDBOptions(dbs: ColorDBInfo[]): SelectOption[] {
    return dbs.map((db) => {
      const suffix = db.source === 'session' ? t('param.sessionSuffix') : ''
      return {
        label:
          t('param.dbOptionLabel', {
            name: db.name,
            entries: db.num_entries,
            channels: db.num_channels,
          }) + suffix,
        value: db.name,
      }
    })
  }
  const appStore = useAppStore()
  const { params: modelValue, inputType, imageDimensions, colordbVersion } = storeToRefs(appStore)

  const loading = ref(true)
  const error = ref<string | null>(null)
  const colorDBs = ref<ColorDBInfo[]>([])
  const defaults = ref<DefaultConfig | null>(null)
  const mode = ref<'simple' | 'advanced'>('simple')
  const selectedMaterial = ref('PLA')
  const selectedVendor = ref('BambooLab')
  const targetWidthMm = ref(200)
  const targetHeightMm = ref(200)
  const selectedPixelPresetIndex = ref(2)
  const customPixelMm = ref(0.1)
  const selectedChannelKeys = ref<string[]>([])
  type ClusterMode = 'off' | 'auto' | 'manual'
  const clusterMode = ref<ClusterMode>('auto')

  const targetDimensionUpperBound = 512
  const clusterCountUpperBound = 256
  const simpleLabelWidth = 112
  const inlineLabelWidth = 96
  const formatTooltip1Decimal = (value: number) => formatFloat(value, 1)
  const formatTooltip2Decimals = (value: number) => formatFloat(value, 2)

  const isRaster = computed(() => inputType.value === 'raster')
  const isVector = computed(() => inputType.value === 'vector')
  const supportsModelGate = computed(() => isRaster.value || isVector.value)

  const effectivePixelMm = computed(() => {
    const presets = getPixelSizePresets(t)
    const preset = presets[selectedPixelPresetIndex.value]
    return preset && preset.value > 0 ? preset.value : customPixelMm.value
  })

  const isCustomPixel = computed(() => {
    const presets = getPixelSizePresets(t)
    const preset = presets[selectedPixelPresetIndex.value]
    return !preset || preset.value === 0
  })

  const pixelPresetOptions = computed<SelectOption[]>(() =>
    getPixelSizePresets(t).map((preset, index) => ({
      label: preset.label,
      value: index,
    })),
  )

  function update(partial: Partial<ConvertAnyParams>) {
    appStore.setParams({ ...modelValue.value, ...partial })
  }

  watch([targetWidthMm, targetHeightMm, effectivePixelMm], () => {
    if (mode.value === 'simple' && isRaster.value) {
      update({
        target_width_mm: targetWidthMm.value,
        target_height_mm: targetHeightMm.value,
        pixel_mm: roundTo(effectivePixelMm.value, 2),
        max_width: 0,
        max_height: 0,
        scale: 1.0,
      })
    } else if (mode.value === 'simple' && isVector.value) {
      update({
        target_width_mm: targetWidthMm.value,
        target_height_mm: targetHeightMm.value,
      })
    }
  })

  const simpleOutputInfo = computed(() => {
    if (!isRaster.value) return null
    const px = effectivePixelMm.value
    if (px <= 0) return null
    const maxPxW = Math.floor(targetWidthMm.value / px)
    const maxPxH = Math.floor(targetHeightMm.value / px)

    const dims = imageDimensions.value
    let actualPxW = maxPxW
    let actualPxH = maxPxH
    let upscaleRatio: number | null = null

    if (dims && dims.width > 0 && dims.height > 0) {
      const scale = Math.min(maxPxW / dims.width, maxPxH / dims.height)
      actualPxW = Math.round(dims.width * scale)
      actualPxH = Math.round(dims.height * scale)
      if (scale > 1.0) upscaleRatio = scale
    }

    return {
      maxPxW,
      maxPxH,
      actualPxW,
      actualPxH,
      actualWidthMm: +(actualPxW * px).toFixed(1),
      actualHeightMm: +(actualPxH * px).toFixed(1),
      upscaleRatio,
    }
  })

  watch(mode, (newMode) => {
    if (newMode === 'simple') {
      const base: Partial<ConvertAnyParams> = {
        target_width_mm: targetWidthMm.value,
        target_height_mm: targetHeightMm.value,
      }
      if (isRaster.value) {
        Object.assign(base, {
          pixel_mm: roundTo(effectivePixelMm.value, 2),
          max_width: 0,
          max_height: 0,
          scale: 1.0,
        })
      }
      update(base)
      return
    }

    if (isRaster.value) {
      update({
        target_width_mm: 0,
        target_height_mm: 0,
      })
    }
  })

  watch(
    () => inputType.value,
    (newType) => {
      if (newType === 'vector') {
        update({
          tessellation_tolerance_mm: modelValue.value.tessellation_tolerance_mm ?? 0.03,
        })
      }
      if (mode.value === 'simple') {
        update({
          target_width_mm: targetWidthMm.value,
          target_height_mm: targetHeightMm.value,
        })
      }
    },
  )

  watch(
    () => ({ type: inputType.value, dither: modelValue.value.dither ?? 'none' }),
    ({ type, dither }) => {
      if (type !== 'raster') return
      if (dither === 'blue_noise') {
        update({ dither: 'none' })
      }
    },
    { immediate: true },
  )

  const availableChannels = computed<ChannelOption[]>(() => {
    const selectedNames = modelValue.value.db_names ?? []
    const seen = new Set<string>()
    const channels: ChannelOption[] = []
    for (const db of colorDBs.value) {
      if (!selectedNames.includes(db.name)) continue
      for (const channel of db.palette) {
        const key = channelKey(channel)
        if (seen.has(key)) continue
        seen.add(key)
        channels.push({ key, color: channel.color, material: channel.material })
      }
    }
    return channels
  })

  const isAllChannelsSelected = computed(
    () =>
      availableChannels.value.length > 0 &&
      selectedChannelKeys.value.length === availableChannels.value.length,
  )

  function emitChannelUpdate() {
    const allKeys = new Set(availableChannels.value.map((channel) => channel.key))
    const selectedSet = new Set(selectedChannelKeys.value)
    if (
      selectedChannelKeys.value.length >= availableChannels.value.length &&
      [...allKeys].every((key) => selectedSet.has(key))
    ) {
      update({ allowed_channels: undefined })
      return
    }

    const channels = availableChannels.value
      .filter((channel) => selectedSet.has(channel.key))
      .map((channel) => ({ color: channel.color, material: channel.material }))
    update({ allowed_channels: channels })
  }

  function handleChannelKeysChange(keys: string[]) {
    selectedChannelKeys.value = keys
    emitChannelUpdate()
  }

  const applicablePresets = computed(() => {
    const availableColors = new Set(availableChannels.value.map((channel) => channel.color))
    return channelPresets.filter((preset) => {
      if (!preset.colors) return true
      return preset.colors.every((color) => availableColors.has(color))
    })
  })

  const activeChannelPreset = computed<string | null>(() => {
    if (isAllChannelsSelected.value) return 'all'
    const selectedColors = new Set(
      availableChannels.value
        .filter((channel) => selectedChannelKeys.value.includes(channel.key))
        .map((channel) => channel.color),
    )
    for (const preset of channelPresets) {
      if (!preset.colors) continue
      if (
        preset.colors.length === selectedColors.size &&
        preset.colors.every((color) => selectedColors.has(color))
      ) {
        return preset.value
      }
    }
    return null
  })

  const clusterCountValue = computed<number>({
    get: () => {
      if (clusterMode.value !== 'manual') return 0
      const raw = modelValue.value.cluster_count ?? 64
      if (!Number.isFinite(raw)) return 64
      return Math.min(clusterCountUpperBound, Math.max(2, Math.round(raw)))
    },
    set: (next) => {
      update({ cluster_count: Math.min(clusterCountUpperBound, Math.max(2, Math.round(next))) })
    },
  })

  function setClusterCount(value: number | null) {
    clusterCountValue.value = value ?? 64
  }

  const lastManualClusterCount = ref(64)

  watch(clusterMode, (mode) => {
    if (mode === 'off') {
      const cur = modelValue.value.cluster_count ?? 64
      if (cur >= 2) lastManualClusterCount.value = cur
      update({ cluster_count: 1 })
    } else if (mode === 'auto') {
      const cur = modelValue.value.cluster_count ?? 64
      if (cur >= 2) lastManualClusterCount.value = cur
      update({ cluster_count: 0 })
    } else {
      update({ cluster_count: lastManualClusterCount.value })
    }
  })

  function applyChannelPreset(value: string) {
    const preset = channelPresets.find((item) => item.value === value)
    if (!preset) return
    if (!preset.colors) {
      selectedChannelKeys.value = availableChannels.value.map((channel) => channel.key)
    } else {
      const target = new Set(preset.colors)
      selectedChannelKeys.value = availableChannels.value
        .filter((channel) => target.has(channel.color))
        .map((channel) => channel.key)
    }
    emitChannelUpdate()
  }

  watch(availableChannels, (newChannels, oldChannels) => {
    const oldKeys = new Set((oldChannels ?? []).map((channel) => channel.key))
    const newKeys = new Set(newChannels.map((channel) => channel.key))

    if (oldKeys.size === newKeys.size && [...oldKeys].every((key) => newKeys.has(key))) return

    let updated = selectedChannelKeys.value.filter((key) => newKeys.has(key))
    for (const channel of newChannels) {
      if (!oldKeys.has(channel.key) && !updated.includes(channel.key)) {
        updated.push(channel.key)
      }
    }

    if (updated.length === 0 && newChannels.length > 0) {
      updated = newChannels.map((channel) => channel.key)
    }

    selectedChannelKeys.value = updated
    emitChannelUpdate()
  })

  const materialOptions = computed<SelectOption[]>(() => {
    const materials = [
      ...new Set(
        colorDBs.value
          .filter((db) => db.source !== 'session' && db.material_type)
          .map((db) => db.material_type!),
      ),
    ]
    materials.sort()
    return materials.map((material) => ({ label: material, value: material }))
  })

  const vendorOptionsForMaterial = computed<SelectOption[]>(() => {
    const vendors = [
      ...new Set(
        colorDBs.value
          .filter(
            (db) =>
              db.source !== 'session' && db.material_type === selectedMaterial.value && db.vendor,
          )
          .map((db) => db.vendor!),
      ),
    ]
    vendors.sort()
    return vendors.map((vendor) => ({ label: vendor, value: vendor }))
  })

  const filteredDBs = computed(() =>
    colorDBs.value.filter((db) => {
      if (db.source === 'session') return true
      if (db.material_type !== selectedMaterial.value) return false
      return db.vendor === selectedVendor.value
    }),
  )

  const isLuminaPreset = computed(
    () => !(selectedMaterial.value === 'PLA' && selectedVendor.value === 'BambooLab'),
  )

  const filteredDBOptions = computed<SelectOption[]>(() => buildDBOptions(filteredDBs.value))
  const modelPackAvailable = computed(
    () => selectedMaterial.value === 'PLA' && selectedVendor.value === 'BambooLab',
  )

  function selectAllFilteredDBs() {
    update({ db_names: filteredDBs.value.map((db) => db.name) })
  }

  watch(selectedMaterial, () => {
    const available = vendorOptionsForMaterial.value.map((option) => option.value as string)
    if (!available.includes(selectedVendor.value)) {
      selectedVendor.value = available[0] ?? ''
    }
    selectAllFilteredDBs()
  })

  watch(selectedVendor, () => {
    selectAllFilteredDBs()
  })

  watch(modelPackAvailable, (available) => {
    if (supportsModelGate.value) {
      update({ model_enable: available })
    }
  })

  function resetParams() {
    if (!defaults.value) return
    targetWidthMm.value = 200
    targetHeightMm.value = 200
    mode.value = 'simple'
    clusterMode.value = 'auto'
    selectedPixelPresetIndex.value = 2
    customPixelMm.value = 0.1
    const currentDbNames = modelValue.value.db_names ?? filteredDBs.value.map((db) => db.name)
    appStore.setParams(
      createInitialConvertParams({
        defaults: defaults.value,
        targetWidthMm: targetWidthMm.value,
        targetHeightMm: targetHeightMm.value,
        pixelMm: effectivePixelMm.value,
        dbNames: currentDbNames,
      }),
    )
    selectedChannelKeys.value = availableChannels.value.map((channel) => channel.key)
  }

  async function loadColorDBs() {
    try {
      colorDBs.value = await fetchAvailableColorDBs()
    } catch {
      // Keep existing options on refresh failure
    }
  }

  watch(
    () => colordbVersion.value,
    () => {
      void loadColorDBs()
    },
  )

  onMounted(async () => {
    try {
      const bootstrap = await fetchParamPanelBootstrap()
      defaults.value = bootstrap.defaults
      colorDBs.value = bootstrap.colorDBs

      const globalDBs = bootstrap.colorDBs.filter((db) => db.source !== 'session')
      const materials = [...new Set(globalDBs.map((db) => db.material_type).filter(Boolean))]
      if (materials.includes('PLA')) {
        selectedMaterial.value = 'PLA'
      } else if (materials.length > 0) {
        selectedMaterial.value = materials[0]!
      }

      const vendors = [
        ...new Set(
          globalDBs
            .filter((db) => db.material_type === selectedMaterial.value)
            .map((db) => db.vendor)
            .filter(Boolean),
        ),
      ]
      if (vendors.includes('BambooLab')) {
        selectedVendor.value = 'BambooLab'
      } else if (vendors.length > 0) {
        selectedVendor.value = vendors[0]!
      }

      const initialDbNames = filteredDBs.value.map((db) => db.name)
      appStore.setParams(
        createInitialConvertParams({
          defaults: bootstrap.defaults,
          targetWidthMm: targetWidthMm.value,
          targetHeightMm: targetHeightMm.value,
          pixelMm: effectivePixelMm.value,
          dbNames: initialDbNames,
        }),
      )
    } catch (nextError: unknown) {
      error.value = nextError instanceof Error ? nextError.message : t('param.loadConfigFailed')
    } finally {
      loading.value = false
    }
  })

  return {
    activeChannelPreset,
    applicablePresets,
    clusterMode,
    clusterModeOptions: clusterModeOpts,
    availableChannels,
    clusterCountUpperBound,
    clusterCountValue,
    colorSpaceOptions,
    customPixelMm,
    ditherOptions: ditherOpts,
    error,
    filteredDBOptions,
    formatTooltip1Decimal,
    formatTooltip2Decimals,
    handleChannelKeysChange,
    imageDimensions,
    inlineLabelWidth,
    inputType,
    isAllChannelsSelected,
    isCustomPixel,
    isLuminaPreset,
    isRaster,
    isVector,
    loading,
    materialOptions,
    mode,
    modelPackAvailable,
    modelValue,
    pixelPresetOptions,
    resetParams,
    selectedChannelKeys,
    selectedMaterial,
    selectedPixelPresetIndex,
    selectedVendor,
    setClusterCount,
    simpleLabelWidth,
    simpleOutputInfo,
    supportsModelGate,
    targetDimensionUpperBound,
    targetHeightMm,
    targetWidthMm,
    tooltips: translatedTooltips,
    roundTo,
    update,
    vendorOptionsForMaterial,
    applyChannelPreset,
  }
}
