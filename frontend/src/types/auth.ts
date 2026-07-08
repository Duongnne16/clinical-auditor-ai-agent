export interface RegisterRequest {
  email: string
  password: string
  full_name?: string | null
}

export interface LoginRequest {
  email: string
  password: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
  doctor_id: string
  email: string
  full_name?: string | null
}

export interface CurrentUser {
  id: number
  doctor_id: string
  email: string
  full_name?: string | null
  is_active: boolean
}
