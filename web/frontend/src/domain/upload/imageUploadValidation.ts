import type { ImageDimensions, InputType } from '../../types'
import { getUploadMaxBytes, getUploadMaxPixels } from '../../runtime/env'
import i18n from '../../locales'

const RASTER_EXTENSIONS = new Set(['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'])
const RASTER_MIME_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/bmp',
  'image/x-ms-bmp',
  'image/tiff',
])
const SVG_EXTENSION = 'svg'
const SVG_MIME_TYPE = 'image/svg+xml'

export const RASTER_IMAGE_ACCEPT =
  '.jpg,.jpeg,.png,.bmp,.tif,.tiff,image/jpeg,image/png,image/bmp,image/x-ms-bmp,image/tiff'
export const CONVERT_IMAGE_ACCEPT = `${RASTER_IMAGE_ACCEPT},.svg,image/svg+xml`
export const RASTER_IMAGE_FORMATS_TEXT = 'JPG / PNG / BMP / TIFF'
export const CONVERT_IMAGE_FORMATS_TEXT = `${RASTER_IMAGE_FORMATS_TEXT} / SVG`

export type UploadValidationScene = 'convert' | 'raster-tool'

export interface UploadValidationOptions {
  maxUploadBytes?: number
  maxPixels?: number
}

export type UploadValidationResult =
  | {
      ok: true
      inputType: InputType
      dimensions: ImageDimensions | null
    }
  | {
      ok: false
      message: string
    }

function getFileExtension(fileName: string): string {
  const dot = fileName.lastIndexOf('.')
  if (dot < 0 || dot === fileName.length - 1) return ''
  return fileName.slice(dot + 1).toLowerCase()
}

function isSvgFile(file: File, ext: string): boolean {
  return ext === SVG_EXTENSION || file.type.toLowerCase() === SVG_MIME_TYPE
}

function isRasterFile(file: File, ext: string): boolean {
  const mime = file.type.toLowerCase()
  return RASTER_EXTENSIONS.has(ext) || RASTER_MIME_TYPES.has(mime)
}

export function readRasterImageDimensions(file: File): Promise<ImageDimensions | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      resolve({ width: img.naturalWidth, height: img.naturalHeight })
      URL.revokeObjectURL(url)
    }
    img.onerror = () => {
      resolve(null)
      URL.revokeObjectURL(url)
    }
    img.src = url
  })
}

function formatLimitPixels(maxPixels: number): string {
  return maxPixels.toLocaleString()
}

export async function validateImageUploadFile(
  file: File,
  scene: UploadValidationScene,
  options: UploadValidationOptions = {},
): Promise<UploadValidationResult> {
  const maxUploadBytes = options.maxUploadBytes ?? getUploadMaxBytes()
  const maxPixels = options.maxPixels ?? getUploadMaxPixels()
  const maxUploadMB = (maxUploadBytes / 1024 / 1024).toFixed(0)
  const ext = getFileExtension(file.name)

  if (file.size <= 0) {
    return { ok: false, message: i18n.global.t('imageUpload.validation.empty') }
  }
  if (file.size > maxUploadBytes) {
    return {
      ok: false,
      message: i18n.global.t('imageUpload.validation.tooLarge', { maxMb: maxUploadMB }),
    }
  }

  const inputType: InputType = isSvgFile(file, ext) ? 'vector' : 'raster'

  if (scene === 'raster-tool' && inputType === 'vector') {
    return {
      ok: false,
      message: i18n.global.t('imageUpload.validation.rasterOnly', {
        formats: RASTER_IMAGE_FORMATS_TEXT,
      }),
    }
  }

  if (inputType === 'vector') {
    if (ext !== SVG_EXTENSION && file.type.toLowerCase() !== SVG_MIME_TYPE && file.type !== '') {
      return {
        ok: false,
        message: i18n.global.t('imageUpload.validation.svgUnsupported'),
      }
    }
    return { ok: true, inputType, dimensions: null }
  }

  if (!isRasterFile(file, ext)) {
    return {
      ok: false,
      message: i18n.global.t('imageUpload.validation.formatUnsupported', {
        formats: RASTER_IMAGE_FORMATS_TEXT,
      }),
    }
  }

  const dimensions = await readRasterImageDimensions(file)
  if (!dimensions || dimensions.width <= 0 || dimensions.height <= 0) {
    return {
      ok: false,
      message: i18n.global.t('imageUpload.validation.decodeFailed'),
    }
  }

  const pixels = dimensions.width * dimensions.height
  if (pixels > maxPixels) {
    return {
      ok: false,
      message: i18n.global.t('imageUpload.validation.tooManyPixels', {
        maxPixels: formatLimitPixels(maxPixels),
      }),
    }
  }

  return { ok: true, inputType, dimensions }
}
