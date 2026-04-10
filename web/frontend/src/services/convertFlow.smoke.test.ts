import { createPinia, setActivePinia } from 'pinia'
import { describe, expect, it, vi } from 'vitest'
import { submitConvertTask } from './convertService'
import { useBlobDownload } from '../composables/useBlobDownload'
import { useAppStore } from '../stores/app'

vi.mock('../api/convert', () => ({
  submitConvertRaster: vi.fn(),
  submitConvertVector: vi.fn(),
}))

vi.mock('../runtime/download', () => ({
  downloadFromUrl: vi.fn(),
  downloadObjectUrl: vi.fn(),
}))

import { submitConvertRaster, submitConvertVector } from '../api/convert'
import { downloadFromUrl } from '../runtime/download'

describe('convert flow smoke', () => {
  it('should support upload->params->submit->download happy path', async () => {
    setActivePinia(createPinia())
    const appStore = useAppStore()

    const file = new File(['abc'], 'demo.png', { type: 'image/png' })
    appStore.setSelectedFile(file)
    appStore.setInputType('raster')
    appStore.setParams({
      db_names: ['db-a'],
      color_layers: 5,
      color_space: 'lab',
      target_width_mm: 120,
      target_height_mm: 100,
      k_candidates: 8,
      generate_preview: true,
    })

    vi.mocked(submitConvertRaster).mockResolvedValueOnce({ task_id: 'task-1' })
    vi.mocked(downloadFromUrl).mockResolvedValueOnce()

    const submitResp = await submitConvertTask(
      appStore.selectedFile!,
      appStore.params,
      appStore.inputType,
    )
    expect(submitResp.task_id).toBe('task-1')
    expect(submitConvertRaster).toHaveBeenCalledTimes(1)
    expect(submitConvertVector).not.toHaveBeenCalled()

    const downloader = useBlobDownload()
    await downloader.downloadByUrl('/api/v1/tasks/task-1/artifacts/result', 'task-1.3mf')
    expect(downloadFromUrl).toHaveBeenCalledWith(
      '/api/v1/tasks/task-1/artifacts/result',
      'task-1.3mf',
    )
  })
})
