import { apiFetch } from './client'
import type {
  PrescriptionHistoryDetail,
  PrescriptionHistoryListItem,
  PrescriptionAuditRequest,
  PrescriptionAuditResponse,
} from '../types/prescriptionAudit'

const REQUEST_TIMEOUT_MS = 120_000
const BACKEND_OFF_MESSAGE =
  'Không kết nối được backend. Hãy kiểm tra FastAPI đang chạy tại http://127.0.0.1:8000.'

const extractErrorMessage = (payload: unknown): string | null => {
  if (!payload || typeof payload !== 'object') {
    return null
  }

  const detail = 'detail' in payload ? payload.detail : null
  const errors = 'errors' in payload ? payload.errors : null

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

  if (Array.isArray(errors)) {
    return errors.map(String).join(', ')
  }

  return null
}

export async function auditPrescription(
  request: PrescriptionAuditRequest,
): Promise<PrescriptionAuditResponse> {
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    const response = await apiFetch('/prescriptions/audit', {
      method: 'POST',
      body: JSON.stringify(request),
      signal: controller.signal,
    })

    let payload: unknown = null

    try {
      payload = await response.json()
    } catch {
      payload = null
    }

    if (!response.ok) {
      const message = extractErrorMessage(payload)
      throw new Error(
        message || `API kiểm tra đơn thuốc trả về lỗi HTTP ${response.status}.`,
      )
    }

    if (!payload || typeof payload !== 'object') {
      throw new Error('API trả về dữ liệu không hợp lệ.')
    }

    return payload as PrescriptionAuditResponse
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Yêu cầu kiểm tra đơn thuốc quá thời gian chờ 120 giây.', {
        cause: error,
      })
    }

    if (error instanceof TypeError) {
      throw new Error(BACKEND_OFF_MESSAGE, { cause: error })
    }

    if (error instanceof Error) {
      throw error
    }

    throw new Error('Không thể kiểm tra đơn thuốc do lỗi không xác định.', {
      cause: error,
    })
  } finally {
    window.clearTimeout(timeoutId)
  }
}

type HistoryListOptions = {
  limit?: number
  offset?: number
}

const readJson = async (response: Response): Promise<unknown> => {
  try {
    return await response.json()
  } catch {
    return null
  }
}

const ensureOk = async (
  response: Response,
  fallbackMessage: string,
): Promise<unknown> => {
  const payload = await readJson(response)

  if (!response.ok) {
    const message = extractErrorMessage(payload)
    throw new Error(message || `${fallbackMessage} HTTP ${response.status}.`)
  }

  return payload
}

export async function listPrescriptionHistory({
  limit = 20,
  offset = 0,
}: HistoryListOptions = {}): Promise<PrescriptionHistoryListItem[]> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  })
  const response = await apiFetch(`/prescriptions/history?${params.toString()}`)
  const payload = await ensureOk(
    response,
    'API lá»‹ch sá»­ kiá»ƒm tra Ä‘Æ¡n thuá»‘c tráº£ vá» lá»—i',
  )

  if (!Array.isArray(payload)) {
    throw new Error('API lá»‹ch sá»­ tráº£ vá» dá»¯ liá»‡u khÃ´ng há»£p lá»‡.')
  }

  return payload as PrescriptionHistoryListItem[]
}

export async function getPrescriptionHistory(
  historyId: number,
): Promise<PrescriptionHistoryDetail> {
  const response = await apiFetch(`/prescriptions/history/${historyId}`)
  const payload = await ensureOk(
    response,
    'API chi tiáº¿t lá»‹ch sá»­ kiá»ƒm tra Ä‘Æ¡n thuá»‘c tráº£ vá» lá»—i',
  )

  if (!payload || typeof payload !== 'object') {
    throw new Error('API chi tiáº¿t lá»‹ch sá»­ tráº£ vá» dá»¯ liá»‡u khÃ´ng há»£p lá»‡.')
  }

  return payload as PrescriptionHistoryDetail
}
