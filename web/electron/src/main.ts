import { app, BrowserWindow, dialog, ipcMain, nativeTheme, shell } from 'electron'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import http from 'node:http'
import net from 'node:net'
import path from 'node:path'
import { ModelDownloader, type ModelDownloadProgress } from './model_downloader'

const DEFAULT_RENDERER_URL = 'http://127.0.0.1:5173'
const DEFAULT_BACKEND_PORT = 18080
const DEFAULT_BACKEND_UPLOAD_MAX_MB = 256
const DEFAULT_BACKEND_MAX_PIXELS = 16384 * 16384
const DEFAULT_BACKEND_MAX_RESULT_MB = 2048
const BACKEND_HEALTH_TIMEOUT_MS = 45_000
const BACKEND_HEALTH_PATH = '/api/v1/health'
const SMOKE_EXIT_ENV = 'CHROMAPRINT3D_ELECTRON_SMOKE_TIMEOUT_MS'
const IPC_GET_API_BASE = 'electron:getApiBase'
const IPC_GET_UPLOAD_LIMITS = 'electron:getUploadLimits'
const IPC_PICK_SINGLE_FILE = 'electron:pickSingleFile'
const IPC_SET_WINDOW_BACKGROUND = 'electron:setWindowBackground'
const IPC_MODELS_GET_STATUS = 'electron:modelsGetStatus'
const IPC_MODELS_CHECK_CONNECTIVITY = 'electron:modelsCheckConnectivity'
const IPC_MODELS_START_DOWNLOAD = 'electron:modelsStartDownload'
const IPC_MODELS_CANCEL_DOWNLOAD = 'electron:modelsCancelDownload'
const IPC_MODELS_RESTART_APP = 'electron:modelsRestartApp'
const IPC_MODELS_PROGRESS_EVENT = 'electron:modelsProgress'
const PACKAGED_BACKEND_DIR = 'backend'
const PACKAGED_FRONTEND_DIR = 'frontend-dist'
const WINDOW_BG_LIGHT = '#F6F8FB'
const WINDOW_BG_DARK = '#171B21'
const DEFAULT_MODEL_DOWNLOAD_RETRIES = 3
const DEFAULT_MODEL_DOWNLOAD_TIMEOUT_MS = 600_000
const DEFAULT_MODEL_DOWNLOAD_BACKOFF_MS = 1500

type RendererTarget = {
  kind: 'url' | 'file'
  value: string
}

type BackendUploadLimits = {
  uploadMaxMb: number
  uploadMaxPixels: number
  maxResultMb: number
}

const MIME_BY_EXTENSION: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
  '.txt': 'text/plain',
}

let mainWindow: BrowserWindow | null = null
let backendProcess: ReturnType<typeof spawn> | null = null
let backendPort = DEFAULT_BACKEND_PORT
let backendUploadLimits: BackendUploadLimits = {
  uploadMaxMb: DEFAULT_BACKEND_UPLOAD_MAX_MB,
  uploadMaxPixels: DEFAULT_BACKEND_MAX_PIXELS,
  maxResultMb: DEFAULT_BACKEND_MAX_RESULT_MB,
}
let isQuitting = false
let modelDownloader: ModelDownloader | null = null

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function repoRoot(): string {
  return path.resolve(__dirname, '..', '..', '..')
}

function runtimeRoot(): string {
  if (app.isPackaged) return process.resourcesPath
  return repoRoot()
}

function resolvePackagedWritableDataDir(): string {
  return path.join(app.getPath('userData'), 'data')
}

function copyDirectoryContents(sourceDir: string, targetDir: string): void {
  if (!fs.existsSync(sourceDir)) {
    throw new Error(`Data source directory not found: ${sourceDir}`)
  }
  fs.mkdirSync(targetDir, { recursive: true })
  const entries = fs.readdirSync(sourceDir, { withFileTypes: true })
  for (const entry of entries) {
    const sourcePath = path.join(sourceDir, entry.name)
    const targetPath = path.join(targetDir, entry.name)
    fs.cpSync(sourcePath, targetPath, {
      recursive: true,
      force: false,
      errorOnExist: false,
    })
  }
}

function initializePackagedDataDir(root: string, dataDir: string): void {
  const bundledDataDir = path.join(root, 'data')
  copyDirectoryContents(bundledDataDir, dataDir)
}

function resolveBackendDataDir(root: string): string {
  const fromEnv = process.env.CHROMAPRINT3D_DATA_DIR?.trim()
  if (fromEnv) return path.resolve(fromEnv)
  if (app.isPackaged) return resolvePackagedWritableDataDir()
  return path.join(root, 'data')
}

function ensureBackendDataDirReady(root: string, dataDir: string): void {
  const hasDataDirOverride = Boolean(process.env.CHROMAPRINT3D_DATA_DIR?.trim())
  if (app.isPackaged && !hasDataDirOverride) {
    initializePackagedDataDir(root, dataDir)
  }
  if (!fs.existsSync(dataDir)) {
    throw new Error(`Backend data directory not found: ${dataDir}`)
  }
}

function parsePort(raw: string | undefined, fallback: number): number {
  if (!raw) return fallback
  const parsed = Number(raw)
  if (!Number.isInteger(parsed) || parsed <= 0 || parsed > 65535) return fallback
  return parsed
}

function parsePositiveInt(raw: string | undefined): number | null {
  if (!raw) return null
  const parsed = Number(raw)
  if (!Number.isInteger(parsed) || parsed <= 0) return null
  return parsed
}

function resolveBackendUploadLimits(): BackendUploadLimits {
  const uploadMaxMb =
    parsePositiveInt(process.env.CHROMAPRINT3D_MAX_UPLOAD_MB) ?? DEFAULT_BACKEND_UPLOAD_MAX_MB
  const uploadMaxPixels =
    parsePositiveInt(process.env.CHROMAPRINT3D_MAX_PIXELS) ?? DEFAULT_BACKEND_MAX_PIXELS
  const maxResultMb =
    parsePositiveInt(process.env.CHROMAPRINT3D_MAX_RESULT_MB) ?? DEFAULT_BACKEND_MAX_RESULT_MB
  return { uploadMaxMb, uploadMaxPixels, maxResultMb }
}

function resolveRendererTarget(): RendererTarget {
  const fromEnv = process.env.CHROMAPRINT3D_RENDERER_URL?.trim()
  if (fromEnv) {
    return { kind: 'url', value: fromEnv }
  }
  if (!app.isPackaged) {
    return { kind: 'url', value: DEFAULT_RENDERER_URL }
  }
  return {
    kind: 'file',
    value: path.join(process.resourcesPath, PACKAGED_FRONTEND_DIR, 'index.html'),
  }
}

function backendBinaryPath(root: string): string {
  const fromEnv = process.env.CHROMAPRINT3D_BACKEND_PATH?.trim()
  if (fromEnv) return path.resolve(fromEnv)
  const binaryName = process.platform === 'win32' ? 'chromaprint3d_server.exe' : 'chromaprint3d_server'
  if (app.isPackaged) {
    if (process.platform !== 'win32') {
      const wrapperPath = path.join(root, PACKAGED_BACKEND_DIR, 'run_chromaprint3d_server.sh')
      if (fs.existsSync(wrapperPath)) {
        return wrapperPath
      }
    }
    return path.join(root, PACKAGED_BACKEND_DIR, binaryName)
  }
  return path.join(root, 'build', 'bin', binaryName)
}

function backendModelPackPath(dataDir: string): string {
  return (
    process.env.CHROMAPRINT3D_MODEL_PACK_PATH?.trim() ||
    path.join(dataDir, 'model_packs')
  )
}

function buildMissingBackendMessage(binaryPath: string): string {
  if (app.isPackaged) {
    return [
      `Backend binary not found: ${binaryPath}`,
      '',
      `Expected packaged binary at "${PACKAGED_BACKEND_DIR}" under resources.`,
      'You can override path via CHROMAPRINT3D_BACKEND_PATH.',
    ].join('\n')
  }
  return [
    `Backend binary not found: ${binaryPath}`,
    '',
    'Build it first from repository root:',
    '  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release',
    '  cmake --build build --target chromaprint3d_server',
  ].join('\n')
}

function healthCheckStatus(baseUrl: string): Promise<number | null> {
  return new Promise((resolve) => {
    const request = http.get(`${baseUrl}${BACKEND_HEALTH_PATH}`, (response) => {
      response.resume()
      resolve(response.statusCode ?? null)
    })
    request.on('error', () => resolve(null))
    request.setTimeout(1500, () => {
      request.destroy()
      resolve(null)
    })
  })
}

async function waitBackendReady(baseUrl: string, timeoutMs: number): Promise<void> {
  const startedAt = Date.now()
  while (Date.now() - startedAt < timeoutMs) {
    const status = await healthCheckStatus(baseUrl)
    if (status === 200) return
    await sleep(500)
  }
  throw new Error(`Backend health check timeout after ${timeoutMs}ms: ${baseUrl}${BACKEND_HEALTH_PATH}`)
}

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer()
    server.once('error', () => {
      resolve(false)
    })
    server.once('listening', () => {
      server.close(() => resolve(true))
    })
    server.listen(port, '127.0.0.1')
  })
}

async function findAvailablePort(startPort: number): Promise<number> {
  const maxProbe = 20
  for (let offset = 0; offset < maxProbe; offset += 1) {
    const port = startPort + offset
    if (await isPortAvailable(port)) return port
  }
  throw new Error(`No available port found in range ${startPort}-${startPort + maxProbe - 1}`)
}

function buildDialogFilters(accept?: string): Electron.FileFilter[] {
  if (!accept) return []
  const extensions = accept
    .split(',')
    .map((token) => token.trim())
    .filter((token) => token.startsWith('.'))
    .map((ext) => ext.slice(1).toLowerCase())
    .filter(Boolean)
  if (extensions.length === 0) return []
  return [{ name: 'Allowed files', extensions }]
}

function guessMimeType(filePath: string): string {
  return MIME_BY_EXTENSION[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream'
}

function isAllowedExternalUrl(url: string): boolean {
  try {
    const parsed = new URL(url)
    return parsed.protocol === 'https:' || parsed.protocol === 'http:' || parsed.protocol === 'mailto:'
  } catch {
    return false
  }
}

function isAllowedNavigation(target: RendererTarget, url: string): boolean {
  if (target.kind === 'file') {
    return url.startsWith('file://')
  }
  try {
    return new URL(url).origin === new URL(target.value).origin
  } catch {
    return false
  }
}

function applyWindowSecurity(window: BrowserWindow, target: RendererTarget): void {
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  window.webContents.on('will-navigate', (event, targetUrl) => {
    if (!isAllowedNavigation(target, targetUrl)) {
      event.preventDefault()
    }
  })
}

function resolveWindowBackgroundColor(dark: boolean): string {
  return dark ? WINDOW_BG_DARK : WINDOW_BG_LIGHT
}

function resolveModelDownloadRetries(): number {
  return (
    parsePositiveInt(process.env.CHROMAPRINT3D_MODEL_DOWNLOAD_RETRIES) ??
    DEFAULT_MODEL_DOWNLOAD_RETRIES
  )
}

function resolveModelDownloadTimeoutMs(): number {
  return (
    parsePositiveInt(process.env.CHROMAPRINT3D_MODEL_DOWNLOAD_TIMEOUT_MS) ??
    DEFAULT_MODEL_DOWNLOAD_TIMEOUT_MS
  )
}

function resolveModelDownloadBackoffMs(): number {
  return (
    parsePositiveInt(process.env.CHROMAPRINT3D_MODEL_DOWNLOAD_BACKOFF_MS) ??
    DEFAULT_MODEL_DOWNLOAD_BACKOFF_MS
  )
}

function emitModelDownloadProgress(progress: ModelDownloadProgress): void {
  if (!mainWindow || mainWindow.isDestroyed()) return
  mainWindow.webContents.send(IPC_MODELS_PROGRESS_EVENT, progress)
}

function registerIpcHandlers(): void {
  ipcMain.on(IPC_GET_API_BASE, (event) => {
    event.returnValue = `http://127.0.0.1:${backendPort}`
  })

  ipcMain.on(IPC_GET_UPLOAD_LIMITS, (event) => {
    event.returnValue = {
      uploadMaxMb: backendUploadLimits.uploadMaxMb,
      uploadMaxPixels: backendUploadLimits.uploadMaxPixels,
    }
  })

  ipcMain.handle(IPC_SET_WINDOW_BACKGROUND, (_event, dark: boolean) => {
    if (!mainWindow || mainWindow.isDestroyed()) return
    mainWindow.setBackgroundColor(resolveWindowBackgroundColor(Boolean(dark)))
  })

  ipcMain.handle('electron:openExternal', async (_event, url: string) => {
    if (!isAllowedExternalUrl(url)) {
      throw new Error('Unsupported external URL protocol')
    }
    await shell.openExternal(url)
  })

  ipcMain.handle(IPC_PICK_SINGLE_FILE, async (_event, accept?: string) => {
    const options = {
      properties: ['openFile'] as Array<'openFile'>,
      filters: buildDialogFilters(accept),
    }
    const result = mainWindow
      ? await dialog.showOpenDialog(mainWindow, options)
      : await dialog.showOpenDialog(options)
    if (result.canceled || result.filePaths.length === 0) return null
    const selectedPath = result.filePaths[0]
    const content = await fs.promises.readFile(selectedPath)
    return {
      name: path.basename(selectedPath),
      mimeType: guessMimeType(selectedPath),
      bytesBase64: content.toString('base64'),
    }
  })

  ipcMain.handle(IPC_MODELS_GET_STATUS, async () => {
    if (!modelDownloader) {
      throw new Error('Model downloader is not initialized')
    }
    return await modelDownloader.getStatus()
  })

  ipcMain.handle(IPC_MODELS_CHECK_CONNECTIVITY, async () => {
    if (!modelDownloader) {
      throw new Error('Model downloader is not initialized')
    }
    return await modelDownloader.checkConnectivity()
  })

  ipcMain.handle(IPC_MODELS_START_DOWNLOAD, async () => {
    if (!modelDownloader) {
      throw new Error('Model downloader is not initialized')
    }
    return await modelDownloader.downloadAll()
  })

  ipcMain.handle(IPC_MODELS_CANCEL_DOWNLOAD, async () => {
    modelDownloader?.cancel()
    return true
  })

  ipcMain.handle(IPC_MODELS_RESTART_APP, async () => {
    app.relaunch()
    app.exit(0)
  })
}

async function startBackend(root: string, dataDir: string): Promise<void> {
  const binaryPath = backendBinaryPath(root)
  const modelPackPath = backendModelPackPath(dataDir)

  if (!fs.existsSync(binaryPath)) {
    throw new Error(buildMissingBackendMessage(binaryPath))
  }
  if (!fs.existsSync(dataDir)) {
    throw new Error(`Backend data directory not found: ${dataDir}`)
  }
  if (!fs.existsSync(modelPackPath)) {
    console.warn(`Model pack path not found: ${modelPackPath}; model matching will be unavailable`)
  }

  const preferredPort = parsePort(process.env.CHROMAPRINT3D_BACKEND_PORT, DEFAULT_BACKEND_PORT)
  backendPort = await findAvailablePort(preferredPort)
  backendUploadLimits = resolveBackendUploadLimits()

  const args = [
    '--host',
    '127.0.0.1',
    '--port',
    String(backendPort),
    '--data',
    dataDir,
    '--model-pack',
    modelPackPath,
    '--max-upload-mb',
    String(backendUploadLimits.uploadMaxMb),
    '--max-pixels',
    String(backendUploadLimits.uploadMaxPixels),
    '--max-result-mb',
    String(backendUploadLimits.maxResultMb),
  ]
  const child = spawn(binaryPath, args, {
    cwd: root,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  backendProcess = child

  child.stdout.on('data', (chunk) => {
    const text = String(chunk).trimEnd()
    if (text) console.log(`[backend] ${text}`)
  })
  child.stderr.on('data', (chunk) => {
    const text = String(chunk).trimEnd()
    if (text) console.error(`[backend] ${text}`)
  })
  child.once('exit', (code, signal) => {
    backendProcess = null
    if (!isQuitting) {
      dialog.showErrorBox(
        'ChromaPrint3D Backend Exited',
        `Backend process exited unexpectedly (code: ${code ?? 'null'}, signal: ${signal ?? 'null'}).`,
      )
      app.quit()
    }
  })
  child.once('error', (error) => {
    console.error('[backend] failed to spawn', error)
  })

  const baseUrl = `http://127.0.0.1:${backendPort}`
  await waitBackendReady(baseUrl, BACKEND_HEALTH_TIMEOUT_MS)
  console.log(`[backend] ready at ${baseUrl}`)
}

async function stopBackend(): Promise<void> {
  const child = backendProcess
  if (!child || child.killed) return

  await new Promise<void>((resolve) => {
    const timer = setTimeout(() => {
      if (!child.killed) child.kill('SIGKILL')
      resolve()
    }, 5000)

    child.once('exit', () => {
      clearTimeout(timer)
      resolve()
    })

    child.kill('SIGTERM')
  })
}

async function createMainWindow(): Promise<void> {
  const preloadPath = path.join(__dirname, 'preload.js')
  const apiBase = `http://127.0.0.1:${backendPort}`
  const target = resolveRendererTarget()
  const window = new BrowserWindow({
    width: 1360,
    height: 900,
    minWidth: 1100,
    minHeight: 760,
    backgroundColor: resolveWindowBackgroundColor(nativeTheme.shouldUseDarkColors),
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      additionalArguments: [`--chromaprint3d-api-base=${apiBase}`],
    },
  })
  mainWindow = window
  window.on('closed', () => {
    mainWindow = null
  })
  applyWindowSecurity(window, target)

  if (target.kind === 'url') {
    await window.loadURL(target.value)
    return
  }
  if (!fs.existsSync(target.value)) {
    throw new Error(`Renderer entry not found: ${target.value}`)
  }
  await window.loadFile(target.value)
}

async function bootstrap(): Promise<void> {
  const root = runtimeRoot()
  const dataDir = resolveBackendDataDir(root)
  ensureBackendDataDirReady(root, dataDir)
  modelDownloader = new ModelDownloader({
    dataDir,
    retriesPerSource: resolveModelDownloadRetries(),
    requestTimeoutMs: resolveModelDownloadTimeoutMs(),
    retryBackoffBaseMs: resolveModelDownloadBackoffMs(),
    onProgress: emitModelDownloadProgress,
  })

  registerIpcHandlers()
  await startBackend(root, dataDir)
  await createMainWindow()

  const smokeExitMs = parsePositiveInt(process.env[SMOKE_EXIT_ENV])
  if (smokeExitMs) {
    setTimeout(() => app.quit(), smokeExitMs).unref()
  }
}

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message
  return String(error)
}

app.on('before-quit', () => {
  isQuitting = true
})

app.on('window-all-closed', () => {
  app.quit()
})

app.on('will-quit', () => {
  void stopBackend()
})

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => {
    app.quit()
  })
}

const hasSingleInstanceLock = app.requestSingleInstanceLock()
if (!hasSingleInstanceLock) {
  app.quit()
  process.exit(0)
}

app.on('second-instance', () => {
  if (!mainWindow) return
  if (mainWindow.isMinimized()) mainWindow.restore()
  mainWindow.focus()
})

app
  .whenReady()
  .then(() => bootstrap())
  .catch((error) => {
    const message = formatError(error)
    console.error('[electron] startup failed:', message)
    dialog.showErrorBox('ChromaPrint3D Electron Startup Failed', message)
    app.quit()
  })
