import type { MachineCatalogResponse } from '../types'
import { request } from './base'

/// Fetch the registered machine catalog.
/// Returns `{ default_machine: '', machines: [] }` when the backend has no
/// catalog loaded (e.g. data dir missing).
export async function fetchMachines(): Promise<MachineCatalogResponse> {
  return request<MachineCatalogResponse>('/api/v1/machines')
}
