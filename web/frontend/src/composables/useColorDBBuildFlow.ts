import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import type { UploadFileInfo } from 'naive-ui'
import { buildColorDB } from '../api/calibration'
import type { ColorDBInfo } from '../types'
import { isValidColorDBName } from './colordbName'

type UseColorDBBuildFlowOptions = {
  onBuilt?: (result: ColorDBInfo) => void | Promise<void>
}

export function useColorDBBuildFlow(options: UseColorDBBuildFlowOptions = {}) {
  const { t } = useI18n()
  const calibImage = ref<File | null>(null)
  const calibMeta = ref<File | null>(null)
  const dbName = ref('')
  const building = ref(false)
  const builtDB = ref<ColorDBInfo | null>(null)
  const buildError = ref<string | null>(null)

  const dbNameError = computed(() => {
    const value = dbName.value.trim()
    if (!value) return t('colordb.build.nameRequired')
    if (!isValidColorDBName(value)) return t('colordb.build.nameInvalid')
    return ''
  })

  const canBuild = computed(
    () => Boolean(calibImage.value && calibMeta.value) && dbNameError.value.length === 0,
  )

  function handleImageUpload({ file }: { file: UploadFileInfo }) {
    calibImage.value = file.file ?? null
    return false
  }

  function handleMetaUpload({ file }: { file: UploadFileInfo }) {
    calibMeta.value = file.file ?? null
    return false
  }

  async function handleBuild(): Promise<ColorDBInfo | null> {
    if (!canBuild.value || !calibImage.value || !calibMeta.value) return null

    building.value = true
    builtDB.value = null
    buildError.value = null
    try {
      const result = await buildColorDB(calibImage.value, calibMeta.value, dbName.value.trim())
      builtDB.value = result
      if (options.onBuilt) {
        await options.onBuilt(result)
      }
      return result
    } catch (error: unknown) {
      buildError.value = error instanceof Error ? error.message : String(error)
      return null
    } finally {
      building.value = false
    }
  }

  return {
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
  }
}
