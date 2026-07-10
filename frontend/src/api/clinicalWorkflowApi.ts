import { apiFetch } from './client'
import type { ChatResponse } from '../types/chat'
import type { PatientContext, PrescriptionAuditResponse } from '../types/prescriptionAudit'

const REQUEST_TIMEOUT_MS = 120_000
const BACKEND_OFF_MESSAGE =
  'Không kết nối được backend. Hãy kiểm tra FastAPI đang chạy tại http://127.0.0.1:8000.'

export type ClinicalWorkflowRequest = {
  input_text: string
  patient_context: PatientContext
  use_gemini: boolean
  query_types: string[]
  top_k_per_type: number
}

export type ClinicalWorkflowResponse = ChatResponse & {
  result_type?: 'audit' | 'chat' | 'refusal' | string
  audit_result?: PrescriptionAuditResponse
  chat_result?: ChatResponse
}

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

export async function runClinicalWorkflow(
  request: ClinicalWorkflowRequest,
): Promise<ClinicalWorkflowResponse> {
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    const response = await apiFetch('/clinical-workflow/run', {
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
        message || `API workflow trả về lỗi HTTP ${response.status}.`,
      )
    }

    if (!payload || typeof payload !== 'object') {
      throw new Error('API workflow trả về dữ liệu không hợp lệ.')
    }

    const workflowResponse = payload as ClinicalWorkflowResponse
    return {
      ...workflowResponse,
      answer: workflowResponse.answer || workflowResponse.message || '',
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Yêu cầu workflow quá thời gian chờ 120 giây.', {
        cause: error,
      })
    }

    if (error instanceof TypeError) {
      throw new Error(BACKEND_OFF_MESSAGE, { cause: error })
    }

    if (error instanceof Error) {
      throw error
    }

    throw new Error('Không thể chạy workflow do lỗi không xác định.', {
      cause: error,
    })
  } finally {
    window.clearTimeout(timeoutId)
  }
}
