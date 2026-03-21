import { submitConvertRaster, submitConvertVector } from '../api/convert'
import { submitConvertRasterMatchOnly, submitConvertVectorMatchOnly } from '../api/recipeEditor'
import { fetchTaskStatus } from '../api/tasks'
import { buildRasterParams, buildVectorParams } from '../domain/params/convertParamBuilders'
import type { ConvertAnyParams, InputType, TaskStatus } from '../types'

export async function submitConvertTask(
  file: File,
  params: ConvertAnyParams,
  inputType: InputType,
): Promise<{ task_id: string }> {
  if (inputType === 'vector') {
    return submitConvertVector(file, buildVectorParams(params))
  }
  return submitConvertRaster(file, buildRasterParams(params))
}

export async function submitMatchOnlyTask(
  file: File,
  params: ConvertAnyParams,
  inputType: InputType = 'raster',
): Promise<{ task_id: string }> {
  if (inputType === 'vector') {
    return submitConvertVectorMatchOnly(file, buildVectorParams(params))
  }
  return submitConvertRasterMatchOnly(file, buildRasterParams(params))
}

export async function fetchConvertTaskStatus(taskId: string): Promise<TaskStatus> {
  return fetchTaskStatus(taskId)
}
