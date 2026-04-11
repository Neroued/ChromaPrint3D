export {}

declare global {
  const __APP_VERSION__: string
  type ElectronEnvApi = {
    apiBase?: string
    uploadMaxMb?: number
    uploadMaxPixels?: number
    updateManifestUrl?: string
  }

  type ElectronStorageApi = {
    getItem?: (key: string) => Promise<string | null>
    setItem?: (key: string, value: string) => Promise<void>
  }

  type ElectronThemeApi = {
    getSystemDarkMode?: () => Promise<boolean>
    setWindowBackground?: (dark: boolean) => Promise<void>
  }

  type ElectronDownloadApi = {
    openExternal?: (url: string) => Promise<void>
    saveObjectUrlAs?: (url: string, filename: string) => Promise<void>
  }

  type ElectronFileApi = {
    pickSingleFile?: (accept?: string) => Promise<File | null>
  }

  type ElectronModelSource = {
    name: string
    baseUrl: string
  }

  type ElectronModelFileStatus = {
    path: string
    sizeBytes: number
    expectedSha256: string
    state: 'installed' | 'missing' | 'invalid'
    exists: boolean
    message?: string
  }

  type ElectronModelDownloadStatus = {
    repository: string
    dataDir: string
    modelDir: string
    manifestPath: string
    sources: ElectronModelSource[]
    models: ElectronModelFileStatus[]
    totalModels: number
    installedModels: number
    missingModels: number
    invalidModels: number
    running: boolean
    currentFilePath?: string
    lastError?: string
  }

  type ElectronModelDownloadProgress = {
    type:
      | 'start'
      | 'file-start'
      | 'file-progress'
      | 'retry'
      | 'source-fallback'
      | 'completed'
      | 'cancelled'
      | 'error'
    message: string
    downloadedBytes: number
    totalBytes: number
    percent: number
    filePath?: string
    fileDownloadedBytes?: number
    fileTotalBytes?: number
    sourceName?: string
    sourceUrl?: string
    attempt?: number
    speedBytesPerSec?: number
  }

  type ElectronModelSourceConnectivity = {
    name: string
    baseUrl: string
    ok: boolean
    checkedModels: number
    reachableModels: number
    statusCode?: number
    responseTimeMs: number
    message: string
  }

  type ElectronModelConnectivityReport = {
    repository: string
    checkedAtMs: number
    totalSources: number
    availableSources: number
    checkedModels: number
    sources: ElectronModelSourceConnectivity[]
  }

  type ElectronModelsApi = {
    getStatus?: () => Promise<ElectronModelDownloadStatus>
    checkConnectivity?: () => Promise<ElectronModelConnectivityReport>
    startDownload?: () => Promise<ElectronModelDownloadStatus>
    cancelDownload?: () => Promise<boolean>
    restartApp?: () => Promise<void>
    onProgress?: (listener: (payload: ElectronModelDownloadProgress) => void) => void
    clearProgressListener?: () => void
  }

  type ElectronRendererApi = {
    env?: ElectronEnvApi
    storage?: ElectronStorageApi
    theme?: ElectronThemeApi
    download?: ElectronDownloadApi
    file?: ElectronFileApi
    models?: ElectronModelsApi
  }

  interface Window {
    electron?: ElectronRendererApi
    umami?: {
      track(): void
      track(event: string, data?: Record<string, unknown>): void
      track(callback: () => { name: string; data?: Record<string, unknown> }): void
      track(callback: (props: Record<string, unknown>) => Record<string, unknown>): void
    }
  }

  interface ImportMetaEnv {
    readonly VITE_UMAMI_HOST?: string
    readonly VITE_UMAMI_WEBSITE_ID?: string
  }
}
