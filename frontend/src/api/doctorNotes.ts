import type { DoctorMemoryNote, DoctorNoteCreatePayload } from '../types/doctorNote'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1'

const extractErrorMessage = (payload: unknown): string | null => {
  if (!payload || typeof payload !== 'object') {
    return null
  }

  const detail = 'detail' in payload ? payload.detail : null
  if (typeof detail === 'string') {
    return detail
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === 'string') {
          return item
        }
        if (item && typeof item === 'object' && 'msg' in item) {
          return String(item.msg)
        }
        return null
      })
      .filter(Boolean)
      .join(', ')
  }

  return null
}

export async function createDoctorNote(
  payload: DoctorNoteCreatePayload,
): Promise<DoctorMemoryNote> {
  const response = await fetch(`${API_BASE_URL}/doctor-notes`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  let responsePayload: unknown
  try {
    responsePayload = await response.json()
  } catch {
    responsePayload = null
  }

  if (!response.ok) {
    throw new Error(
      extractErrorMessage(responsePayload) ||
        `API lưu ghi chú trả về lỗi HTTP ${response.status}.`,
    )
  }

  return responsePayload as DoctorMemoryNote
}

export async function searchDoctorNotes(
  query: string,
  topK = 5,
): Promise<DoctorMemoryNote[]> {
  const params = new URLSearchParams({
    q: query,
    top_k: String(topK),
  })
  const response = await fetch(`${API_BASE_URL}/doctor-notes/search?${params}`)

  if (!response.ok) {
    return []
  }

  const payload = (await response.json()) as unknown
  return Array.isArray(payload) ? (payload as DoctorMemoryNote[]) : []
}
