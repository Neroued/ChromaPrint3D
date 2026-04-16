import type { AnnouncementsResponse } from '../types'
import { request } from './base'

export async function fetchAnnouncements(): Promise<AnnouncementsResponse> {
  return request<AnnouncementsResponse>('/api/v1/announcements')
}
