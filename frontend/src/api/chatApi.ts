import { apiFetch } from './client'
import type { ChatResponse } from '../types/chat'

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

export async function sendChatMessage(message: string): Promise<ChatResponse> {
  const trimmedMessage = message.trim()

  if (!trimmedMessage) {
    throw new Error('Vui lòng nhập câu hỏi trước khi gửi.')
  }

  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    const response = await apiFetch('/chat', {
      method: 'POST',
      body: JSON.stringify({ message: trimmedMessage }),
      signal: controller.signal,
    })

    let payload: unknown = null

    try {
      payload = await response.json()
    } catch {
      payload = null
    }

    if (!response.ok) {
      const errorMessage = extractErrorMessage(payload)
      throw new Error(
        errorMessage || `API hỏi về thuốc trả về lỗi HTTP ${response.status}.`,
      )
    }

    if (!payload || typeof payload !== 'object') {
      throw new Error('API trả về dữ liệu không hợp lệ.')
    }

    const chatResponse = payload as ChatResponse
    return {
      ...chatResponse,
      answer: chatResponse.answer || chatResponse.message || '',
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Yêu cầu hỏi về thuốc quá thời gian chờ 120 giây.', {
        cause: error,
      })
    }

    if (error instanceof TypeError) {
      throw new Error(BACKEND_OFF_MESSAGE, { cause: error })
    }

    if (error instanceof Error) {
      throw error
    }

    throw new Error('Không thể hỏi về thuốc do lỗi không xác định.', {
      cause: error,
    })
  } finally {
    window.clearTimeout(timeoutId)
  }
}
