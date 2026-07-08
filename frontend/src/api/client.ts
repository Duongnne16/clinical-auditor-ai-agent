const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000/api/v1'

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/+$/, '')

export const ACCESS_TOKEN_STORAGE_KEY = 'clinical_auditor_access_token'

const canUseLocalStorage = (): boolean =>
  typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'

export function getAccessToken(): string | null {
  if (!canUseLocalStorage()) {
    return null
  }

  return window.localStorage.getItem(ACCESS_TOKEN_STORAGE_KEY)
}

export function setAccessToken(token: string): void {
  if (!canUseLocalStorage()) {
    return
  }

  window.localStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token)
}

export function clearAccessToken(): void {
  if (!canUseLocalStorage()) {
    return
  }

  window.localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY)
}

const notifyUnauthorized = (): void => {
  if (typeof window === 'undefined') {
    return
  }

  window.dispatchEvent(new Event('auth:unauthorized'))
}

const resolveApiUrl = (input: RequestInfo | URL): RequestInfo | URL => {
  if (typeof input !== 'string') {
    return input
  }

  if (!input.startsWith('/')) {
    return input
  }

  return `${API_BASE_URL}${input}`
}

const shouldSetJsonContentType = (init?: RequestInit): boolean => {
  if (!init?.body) {
    return false
  }

  if (typeof FormData !== 'undefined' && init.body instanceof FormData) {
    return false
  }

  return true
}

export async function apiFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers)
  const token = getAccessToken()

  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  if (shouldSetJsonContentType(init) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(resolveApiUrl(input), {
    ...init,
    headers,
  })

  if (response.status === 401) {
    clearAccessToken()
    notifyUnauthorized()
  }

  return response
}
