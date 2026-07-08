import { apiFetch, clearAccessToken, setAccessToken } from './client'
import type {
  CurrentUser,
  LoginRequest,
  RegisterRequest,
  TokenResponse,
} from '../types/auth'

const extractErrorMessage = (payload: unknown): string | null => {
  if (!payload || typeof payload !== 'object') {
    return null
  }

  const detail = 'detail' in payload ? payload.detail : null
  const message = 'message' in payload ? payload.message : null

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

  if (typeof message === 'string') {
    return message
  }

  return null
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
    throw new Error(
      extractErrorMessage(payload) ||
        `${fallbackMessage} HTTP ${response.status}.`,
    )
  }

  return payload
}

export async function register(
  payload: RegisterRequest,
): Promise<TokenResponse> {
  const response = await apiFetch('/auth/register', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  const tokenResponse = (await ensureOk(
    response,
    'API đăng ký trả về lỗi',
  )) as TokenResponse

  setAccessToken(tokenResponse.access_token)
  return tokenResponse
}

export async function login(payload: LoginRequest): Promise<TokenResponse> {
  const response = await apiFetch('/auth/login', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  const tokenResponse = (await ensureOk(
    response,
    'API đăng nhập trả về lỗi',
  )) as TokenResponse

  setAccessToken(tokenResponse.access_token)
  return tokenResponse
}

export async function getCurrentUser(): Promise<CurrentUser> {
  const response = await apiFetch('/auth/me')
  return (await ensureOk(
    response,
    'API lấy thông tin người dùng trả về lỗi',
  )) as CurrentUser
}

export function logout(): void {
  clearAccessToken()
}
