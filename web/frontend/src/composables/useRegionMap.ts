import { ref, type Ref } from 'vue'
import { fetchBlobWithSession } from '../runtime/protectedRequest'
import { getRegionMapPath } from '../api/recipeEditor'
import { buildApiUrl } from '../runtime/env'

export interface RegionMapData {
  width: number
  height: number
  regionIds: Uint32Array
}

export function useRegionMap() {
  const regionMap = ref<RegionMapData | null>(null) as Ref<RegionMapData | null>
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function load(taskId: string, width: number, height: number) {
    loading.value = true
    error.value = null
    try {
      const blob = await fetchBlobWithSession(buildApiUrl(getRegionMapPath(taskId)))
      const buf = await blob.arrayBuffer()
      const expectedBytes = width * height * 4
      if (buf.byteLength !== expectedBytes) {
        throw new Error(
          `Region map size mismatch: got ${buf.byteLength}, expected ${expectedBytes}`,
        )
      }
      regionMap.value = {
        width,
        height,
        regionIds: new Uint32Array(buf),
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
    if (x < 0 || x >= map.width || y < 0 || y >= map.height) return null
    return map.regionIds[y * map.width + x] ?? null
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
