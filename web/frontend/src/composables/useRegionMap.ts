import { ref, type Ref } from 'vue'
import { fetchWithSession } from '../runtime/protectedRequest'
import { getRegionMapPath } from '../api/recipeEditor'
import { buildApiUrl } from '../runtime/env'

export interface RegionMapData {
  width: number
  height: number
  regionIds: Uint32Array
  sourceWidth: number
  sourceHeight: number
}

export function useRegionMap() {
  const regionMap = ref<RegionMapData | null>(null) as Ref<RegionMapData | null>
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function load(taskId: string, width: number, height: number) {
    loading.value = true
    error.value = null
    try {
      const response = await fetchWithSession(buildApiUrl(getRegionMapPath(taskId)))
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const buf = await response.arrayBuffer()

      let mapW = width
      let mapH = height
      const hdrW = response.headers.get('X-Region-Map-Width')
      const hdrH = response.headers.get('X-Region-Map-Height')
      if (hdrW && hdrH) {
        mapW = parseInt(hdrW, 10)
        mapH = parseInt(hdrH, 10)
      }

      const expectedBytes = mapW * mapH * 4
      if (buf.byteLength !== expectedBytes) {
        throw new Error(
          `Region map size mismatch: got ${buf.byteLength}, expected ${expectedBytes}`,
        )
      }
      regionMap.value = {
        width: mapW,
        height: mapH,
        regionIds: new Uint32Array(buf),
        sourceWidth: width,
        sourceHeight: height,
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      regionMap.value = null
    } finally {
      loading.value = false
    }
  }

  function getRegionAtPixel(x: number, y: number): number | null {
    const map = regionMap.value
    if (!map) return null
    const mx = Math.floor((x * map.width) / map.sourceWidth)
    const my = Math.floor((y * map.height) / map.sourceHeight)
    if (mx < 0 || mx >= map.width || my < 0 || my >= map.height) return null
    return map.regionIds[my * map.width + mx] ?? null
  }

  function getRegionIdsForRecipeIndex(
    recipeIndex: number,
    regionRecipeIndices: number[],
  ): number[] {
    return regionRecipeIndices.reduce<number[]>((acc, ri, regionId) => {
      if (ri === recipeIndex) acc.push(regionId)
      return acc
    }, [])
  }

  function clear() {
    regionMap.value = null
    error.value = null
  }

  return {
    regionMap,
    loading,
    error,
    load,
    getRegionAtPixel,
    getRegionIdsForRecipeIndex,
    clear,
  }
}
