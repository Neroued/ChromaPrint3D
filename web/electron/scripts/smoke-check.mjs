#!/usr/bin/env node

import { existsSync, readdirSync, statSync } from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import { spawn, spawnSync } from 'node:child_process'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const electronRoot = path.resolve(__dirname, '..')
const resourcesRoot = path.resolve(
  process.env.CHROMAPRINT3D_ELECTRON_RESOURCES_DIR ?? path.join(electronRoot, 'resources'),
)
const backendPlatform = (process.env.CHROMAPRINT3D_BACKEND_PLATFORM ?? process.platform).toLowerCase()

function fail(message) {
  throw new Error(message)
}

function assertPathExists(targetPath, label) {
  if (!existsSync(targetPath)) {
    fail(`Missing ${label}: ${targetPath}`)
  }
}

function backendBinaryPath() {
  const binaryName =
    backendPlatform === 'win32' || backendPlatform === 'windows'
      ? 'chromaprint3d_server.exe'
      : 'chromaprint3d_server'
  return path.join(resourcesRoot, 'backend', binaryName)
}

function backendLaunchPath() {
  if (backendPlatform === 'win32' || backendPlatform === 'windows') {
    return backendBinaryPath()
  }
  const wrapperPath = path.join(resourcesRoot, 'backend', 'run_chromaprint3d_server.sh')
  if (existsSync(wrapperPath)) {
    return wrapperPath
  }
  return backendBinaryPath()
}

function validateResourcesStructure() {
  assertPathExists(path.join(resourcesRoot, 'frontend-dist'), 'frontend-dist directory')
  assertPathExists(path.join(resourcesRoot, 'frontend-dist', 'index.html'), 'frontend-dist/index.html')
  assertPathExists(path.join(resourcesRoot, 'backend'), 'backend directory')
  assertPathExists(path.join(resourcesRoot, 'data'), 'data directory')
  const modelPackDir = path.join(resourcesRoot, 'data', 'model_packs')
  if (existsSync(modelPackDir)) {
    console.log('✅ data/model_packs/ exists')
  } else {
    console.warn('⚠️  data/model_packs/ not found; model matching will be unavailable')
  }
  const dbRoot = path.join(resourcesRoot, 'data', 'dbs')
  assertPathExists(dbRoot, 'data/dbs')
  if (!hasAnyColorDbJson(dbRoot)) {
    fail(`No ColorDB JSON file found under: ${dbRoot}`)
  }

  const backendBinary = backendBinaryPath()
  assertPathExists(backendBinary, 'backend executable')
  if (backendPlatform !== 'win32' && backendPlatform !== 'windows') {
    const mode = statSync(backendBinary).mode
    if ((mode & 0o111) === 0) {
      fail(`Backend executable is not marked executable: ${backendBinary}`)
    }
  }
}

function hasAnyColorDbJson(rootDir) {
  const queue = [rootDir]
  while (queue.length > 0) {
    const current = queue.shift()
    if (!current) continue
    const entries = readdirSync(current, { withFileTypes: true })
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name)
      if (entry.isDirectory()) {
        queue.push(fullPath)
        continue
      }
      if (entry.isFile() && entry.name.toLowerCase().endsWith('.json')) {
        return true
      }
    }
  }
  return false
}

function checkBackendExecutable(binaryPath) {
  const result = spawnSync(binaryPath, ['--help'], {
    stdio: 'pipe',
    windowsHide: true,
    shell: process.platform === 'win32',
  })
  if (result.error) {
    throw result.error
  }
  if (result.status !== 0) {
    const stderr = (result.stderr ?? '').toString()
    fail(`Backend --help check failed (${result.status}): ${stderr}`)
  }
}

function resolveLaunchCommand() {
  const electronCli = path.resolve(electronRoot, 'node_modules', 'electron', 'cli.js')
  if (!existsSync(electronCli)) {
    fail(`electron cli.js not found: ${electronCli}`)
  }
  const nodeArgs = [electronCli, '.']
  if (process.platform === 'linux') {
    const xvfbProbe = spawnSync('xvfb-run', ['--help'], { stdio: 'ignore' })
    if (xvfbProbe.status === 0) {
      return { command: 'xvfb-run', args: ['-a', process.execPath, ...nodeArgs] }
    }
  }
  return { command: process.execPath, args: nodeArgs }
}

async function runElectronStartupCheck() {
  const backendBinary = backendLaunchPath()
  const frontendEntry = path.join(resourcesRoot, 'frontend-dist', 'index.html')
  const modelPackPath = path.join(resourcesRoot, 'data', 'model_packs')
  const smokeExitMs = process.env.CHROMAPRINT3D_ELECTRON_SMOKE_TIMEOUT_MS?.trim() || '4000'
  const timeoutMs = Number(process.env.CHROMAPRINT3D_ELECTRON_SMOKE_TOTAL_TIMEOUT_MS ?? '30000')
  const rendererUrl = pathToFileURL(frontendEntry).toString()

  const env = {
    ...process.env,
    CHROMAPRINT3D_BACKEND_PATH: backendBinary,
    CHROMAPRINT3D_DATA_DIR: path.join(resourcesRoot, 'data'),
    CHROMAPRINT3D_MODEL_PACK_PATH: modelPackPath,
    CHROMAPRINT3D_RENDERER_URL: rendererUrl,
    CHROMAPRINT3D_ELECTRON_SMOKE_TIMEOUT_MS: smokeExitMs,
  }

  const { command, args } = resolveLaunchCommand()
  const child = spawn(command, args, {
    cwd: electronRoot,
    env,
    stdio: 'pipe',
  })

  let output = ''
  child.stdout.on('data', (chunk) => {
    output += String(chunk)
  })
  child.stderr.on('data', (chunk) => {
    output += String(chunk)
  })

  const exitCode = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      reject(new Error(`Electron startup smoke check timed out after ${timeoutMs}ms`))
    }, timeoutMs)

    child.once('error', (error) => {
      clearTimeout(timer)
      reject(error)
    })
    child.once('exit', (code) => {
      clearTimeout(timer)
      resolve(code)
    })
  })

  if (typeof exitCode !== 'number' || exitCode !== 0) {
    fail(`Electron startup smoke check failed with code=${String(exitCode)}\n${output}`)
  }
}

async function main() {
  validateResourcesStructure()
  checkBackendExecutable(backendLaunchPath())
  await runElectronStartupCheck()
  console.log('Electron smoke checks passed.')
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error)
  console.error(message)
  process.exit(1)
})
