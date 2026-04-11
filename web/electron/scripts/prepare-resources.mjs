#!/usr/bin/env node

import { cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import process from 'node:process'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const electronRoot = path.resolve(__dirname, '..')
const repoRoot = path.resolve(electronRoot, '..', '..')

const resourcesRoot = path.resolve(
  process.env.CHROMAPRINT3D_ELECTRON_RESOURCES_DIR ?? path.join(electronRoot, 'resources'),
)
const frontendTargetDir = path.join(resourcesRoot, 'frontend-dist')
const backendTargetDir = path.join(resourcesRoot, 'backend')
const dataTargetDir = path.join(resourcesRoot, 'data')
const dataSourceDir = path.resolve(process.env.CHROMAPRINT3D_DATA_SOURCE_DIR ?? path.join(repoRoot, 'data'))
const backendPlatform = (process.env.CHROMAPRINT3D_BACKEND_PLATFORM ?? process.platform).toLowerCase()
const iconSourceSvg = path.join(repoRoot, 'web', 'frontend', 'public', 'favicon.svg')
const iconOutputDir = path.join(resourcesRoot, 'icons')
const linuxIconSizes = [16, 32, 48, 64, 128, 256, 512, 1024]
const winIconPath = path.join(iconOutputDir, 'icon.ico')
const macIconPath = path.join(iconOutputDir, 'icon.icns')

function fail(message) {
  throw new Error(message)
}

function run(command, args) {
  const result = spawnSync(command, args, { stdio: 'inherit' })
  if (result.error) {
    throw result.error
  }
  if (result.status !== 0) {
    fail(`Command failed (${result.status}): ${command} ${args.join(' ')}`)
  }
}

function runOptional(command, args) {
  const result = spawnSync(command, args, { stdio: 'pipe' })
  return result.status === 0
}

function runOrFail(command, args, context) {
  const result = spawnSync(command, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
  if (result.error) {
    throw result.error
  }
  if (result.status !== 0) {
    const stderr = (result.stderr ?? '').trim()
    const stdout = (result.stdout ?? '').trim()
    fail(
      `${context}\nCommand failed (${result.status}): ${command} ${args.join(' ')}\n` +
        `${stdout ? `stdout: ${stdout}\n` : ''}${stderr ? `stderr: ${stderr}` : ''}`,
    )
  }
}

function runCapture(command, args) {
  const result = spawnSync(command, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
  if (result.error) {
    throw result.error
  }
  if (result.status !== 0) {
    return ''
  }
  return result.stdout ?? ''
}

function ensureCleanDir(dirPath) {
  rmSync(dirPath, { recursive: true, force: true })
  mkdirSync(dirPath, { recursive: true })
}

function copyDirContents(srcDir, destDir) {
  if (!existsSync(srcDir)) {
    fail(`Directory does not exist: ${srcDir}`)
  }
  mkdirSync(destDir, { recursive: true })
  const entries = readdirSync(srcDir, { withFileTypes: true })
  for (const entry of entries) {
    const sourcePath = path.join(srcDir, entry.name)
    const targetPath = path.join(destDir, entry.name)
    cpSync(sourcePath, targetPath, { recursive: true })
  }
}

function resolveFrontendSourceDir() {
  const fromDistEnv = process.env.CHROMAPRINT3D_FRONTEND_DIST_DIR?.trim()
  if (fromDistEnv) return path.resolve(fromDistEnv)
  const fromArtifactEnv = process.env.CHROMAPRINT3D_FRONTEND_ARTIFACT_DIR?.trim()
  if (fromArtifactEnv) return path.resolve(fromArtifactEnv)
  return path.join(repoRoot, 'artifacts', 'frontend-dist')
}

function resolveBackendArchive() {
  const fromEnv = process.env.CHROMAPRINT3D_BACKEND_ARCHIVE?.trim()
  if (fromEnv) return path.resolve(fromEnv)

  const artifactsDir = path.join(repoRoot, 'artifacts')
  if (!existsSync(artifactsDir)) {
    fail('Backend archive not provided and artifacts directory does not exist.')
  }

  const candidates = []
  const firstLevel = readdirSync(artifactsDir, { withFileTypes: true })
  for (const entry of firstLevel) {
    if (!entry.isDirectory() || !entry.name.startsWith('backend-')) continue
    const backendDir = path.join(artifactsDir, entry.name)
    for (const file of readdirSync(backendDir, { withFileTypes: true })) {
      if (!file.isFile()) continue
      if (
        file.name.endsWith('.tar.gz') ||
        file.name.endsWith('.tgz') ||
        file.name.endsWith('.zip') ||
        file.name.endsWith('.tar')
      ) {
        candidates.push(path.join(backendDir, file.name))
      }
    }
  }

  if (candidates.length === 0) {
    fail('No backend archive found under artifacts/backend-* directories.')
  }
  if (candidates.length > 1) {
    fail(
      `Found multiple backend archives. Please set CHROMAPRINT3D_BACKEND_ARCHIVE explicitly.\n${candidates.join('\n')}`,
    )
  }
  return candidates[0]
}

function extractArchive(archivePath, destinationDir) {
  if (!existsSync(archivePath)) {
    fail(`Backend archive does not exist: ${archivePath}`)
  }
  mkdirSync(destinationDir, { recursive: true })

  if (archivePath.endsWith('.tar.gz') || archivePath.endsWith('.tgz')) {
    run('tar', ['-xzf', archivePath, '-C', destinationDir])
    return
  }
  if (archivePath.endsWith('.tar')) {
    run('tar', ['-xf', archivePath, '-C', destinationDir])
    return
  }
  if (archivePath.endsWith('.zip')) {
    if (process.platform === 'win32') {
      const command = `Expand-Archive -LiteralPath '${archivePath.replace(/'/g, "''")}' -DestinationPath '${destinationDir.replace(/'/g, "''")}' -Force`
      run('powershell', ['-NoProfile', '-Command', command])
      return
    }
    run('unzip', ['-q', archivePath, '-d', destinationDir])
    return
  }

  fail(`Unsupported archive format: ${archivePath}`)
}

function parseOtoolDependencies(targetFile) {
  const output = runCapture('otool', ['-L', targetFile])
  if (!output) return []
  return output
    .split('\n')
    .slice(1)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split(' (', 1)[0])
}

function canonicalLibPrefix(fileName) {
  const noExt = fileName.replace(/\.dylib$/i, '')
  return noExt.replace(/\.[0-9].*$/, '')
}

function ensureExpectedMacLibName(libDir, expectedBaseName, fallbackSourcePath) {
  const expectedPath = path.join(libDir, expectedBaseName)
  if (existsSync(expectedPath)) {
    return { ok: true, added: false }
  }

  const expectedPrefix = canonicalLibPrefix(expectedBaseName)
  const dylibs = readdirSync(libDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.dylib'))
    .map((entry) => entry.name)
  const candidates = dylibs.filter((name) => canonicalLibPrefix(name) === expectedPrefix)

  if (candidates.length > 0) {
    const selected = [...candidates].sort((a, b) => a.length - b.length || a.localeCompare(b))[0]
    cpSync(path.join(libDir, selected), expectedPath, { recursive: false })
    return { ok: true, added: true }
  }

  if (fallbackSourcePath && existsSync(fallbackSourcePath)) {
    cpSync(fallbackSourcePath, expectedPath, { recursive: false, dereference: true })
    return { ok: true, added: true }
  }

  return { ok: false, added: false }
}

function normalizeMacBackendDependencies() {
  if (!(backendPlatform === 'mac' || backendPlatform === 'macos' || backendPlatform === 'darwin')) {
    return
  }

  const backendBinary = path.join(backendTargetDir, 'chromaprint3d_server')
  const libDir = path.join(backendTargetDir, 'lib')
  if (!existsSync(backendBinary) || !existsSync(libDir)) {
    return
  }

  const listDylibs = () =>
    readdirSync(libDir, { withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name.endsWith('.dylib'))
      .map((entry) => path.join(libDir, entry.name))

  // Multiple passes allow us to recursively absorb transitive external libs
  // (e.g. OpenBLAS) that are discovered only after first-level copies.
  for (let pass = 0; pass < 4; pass += 1) {
    let addedInPass = false
    const targets = [backendBinary, ...listDylibs()]
    for (const target of targets) {
      const deps = parseOtoolDependencies(target)
      for (const dep of deps) {
        if (dep.startsWith('/System/') || dep.startsWith('/usr/lib/')) {
          continue
        }
        if (
          dep.startsWith('@rpath/') ||
          dep.startsWith('@loader_path/') ||
          dep.startsWith('@executable_path/')
        ) {
          continue
        }
        if (!dep.startsWith('/')) {
          continue
        }

        const depBase = path.basename(dep)
        const ensured = ensureExpectedMacLibName(libDir, depBase, dep)
        if (!ensured.ok) {
          continue
        }
        if (ensured.added) {
          addedInPass = true
        }
        runOrFail(
          'install_name_tool',
          ['-change', dep, `@rpath/${depBase}`, target],
          `Failed to rewrite dependency for ${path.basename(target)}: ${dep}`,
        )
      }
    }
    if (!addedInPass) {
      break
    }
  }

  const dylibs = listDylibs()
  runOptional('install_name_tool', ['-add_rpath', '@executable_path/lib', backendBinary])
  for (const dylibPath of dylibs) {
    const baseName = path.basename(dylibPath)
    runOrFail(
      'install_name_tool',
      ['-id', `@rpath/${baseName}`, dylibPath],
      `Failed to set install_name for ${baseName}`,
    )
    runOptional('install_name_tool', ['-add_rpath', '@loader_path', dylibPath])
  }

  const problematic = []
  const forbiddenPrefixes = ['/usr/local/', '/opt/homebrew/', '/opt/local/']
  for (const target of [backendBinary, ...listDylibs()]) {
    const deps = parseOtoolDependencies(target)
    for (const dep of deps) {
      if (forbiddenPrefixes.some((prefix) => dep.startsWith(prefix))) {
        problematic.push(`${path.basename(target)} -> ${dep}`)
      }
    }
  }
  if (problematic.length > 0) {
    fail(
      `Unresolved absolute macOS dependencies remain after normalization:\n${problematic.join('\n')}`,
    )
  }
}

function assertPreparedResources() {
  const frontendEntry = path.join(frontendTargetDir, 'index.html')
  if (!existsSync(frontendEntry)) {
    fail(`Missing frontend entry: ${frontendEntry}`)
  }

  const backendBinary =
    backendPlatform === 'win32' || backendPlatform === 'windows'
      ? path.join(backendTargetDir, 'chromaprint3d_server.exe')
      : path.join(backendTargetDir, 'chromaprint3d_server')
  if (!existsSync(backendBinary)) {
    fail(`Missing backend executable: ${backendBinary}`)
  }

  const modelPackDir = path.join(dataTargetDir, 'model_packs')
  if (existsSync(modelPackDir)) {
    const msgpackFiles = readdirSync(modelPackDir).filter(f => f.endsWith('.msgpack'))
    if (msgpackFiles.length === 0) {
      console.warn('model_packs/ directory exists but contains no .msgpack files')
    }
  } else {
    console.warn('model_packs/ directory not found; model matching will be unavailable')
  }

  const dbRoot = path.join(dataTargetDir, 'dbs')
  if (!existsSync(dbRoot)) {
    fail(`Missing ColorDB directory: ${dbRoot}`)
  }
  if (!hasAnyColorDbJson(dbRoot)) {
    fail(`No ColorDB JSON file found under: ${dbRoot}`)
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

async function prepareIcons() {
  if (!existsSync(iconSourceSvg)) {
    fail(`Missing icon source file: ${iconSourceSvg}`)
  }

  const sharpModule = await import('sharp')
  const pngToIcoModule = await import('png-to-ico')
  const sharp = sharpModule.default
  const pngToIco = pngToIcoModule.default ?? pngToIcoModule

  ensureCleanDir(iconOutputDir)

  const renderPngFile = (size, filePath) =>
    sharp(iconSourceSvg)
      .resize(size, size, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toFile(filePath)

  const renderPngBuffer = (size) =>
    sharp(iconSourceSvg)
      .resize(size, size, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toBuffer()

  for (const size of linuxIconSizes) {
    await renderPngFile(size, path.join(iconOutputDir, `${size}x${size}.png`))
  }

  const icoSizes = [16, 24, 32, 48, 64, 128, 256]
  const icoBuffers = []
  for (const size of icoSizes) {
    icoBuffers.push(await renderPngBuffer(size))
  }
  const icoBytes = await pngToIco(icoBuffers)
  writeFileSync(winIconPath, icoBytes)

  if (process.platform !== 'darwin') return

  const iconsetRoot = mkdtempSync(path.join(tmpdir(), 'chromaprint3d-iconset-'))
  const iconsetDir = path.join(iconsetRoot, 'icon.iconset')
  mkdirSync(iconsetDir, { recursive: true })
  try {
    const baseSizes = [16, 32, 64, 128, 256, 512]
    for (const size of baseSizes) {
      await renderPngFile(size, path.join(iconsetDir, `icon_${size}x${size}.png`))
      await renderPngFile(size * 2, path.join(iconsetDir, `icon_${size}x${size}@2x.png`))
    }
    runOrFail('iconutil', ['-c', 'icns', iconsetDir, '-o', macIconPath], 'Failed to build .icns icon')
  } finally {
    rmSync(iconsetRoot, { recursive: true, force: true })
  }
}

async function prepare() {
  mkdirSync(resourcesRoot, { recursive: true })
  ensureCleanDir(frontendTargetDir)
  ensureCleanDir(backendTargetDir)
  ensureCleanDir(dataTargetDir)

  const frontendSourceDir = resolveFrontendSourceDir()
  if (!existsSync(path.join(frontendSourceDir, 'index.html'))) {
    fail(`Frontend artifact is invalid (index.html not found): ${frontendSourceDir}`)
  }
  copyDirContents(frontendSourceDir, frontendTargetDir)

  const backendArchive = resolveBackendArchive()
  const unpackDir = mkdtempSync(path.join(tmpdir(), 'chromaprint3d-backend-'))
  try {
    extractArchive(backendArchive, unpackDir)
    copyDirContents(unpackDir, backendTargetDir)
  } finally {
    rmSync(unpackDir, { recursive: true, force: true })
  }

  normalizeMacBackendDependencies()

  copyDirContents(dataSourceDir, dataTargetDir)
  await prepareIcons()
  assertPreparedResources()

  console.log('Prepared Electron resources:')
  console.log(`- frontend: ${frontendTargetDir}`)
  console.log(`- backend : ${backendTargetDir}`)
  console.log(`- data    : ${dataTargetDir}`)
  console.log(`- icons   : ${iconOutputDir}`)
}

prepare().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error))
  process.exit(1)
})
