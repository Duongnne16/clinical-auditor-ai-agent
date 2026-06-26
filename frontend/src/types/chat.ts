export type ChatRole = 'user' | 'assistant'

export type RiskLevel = 'high' | 'moderate' | 'low' | 'unknown'

export type ChatIntent =
  | 'drug_interaction_query'
  | 'single_drug_query'
  | 'out_of_scope'
  | string

export type ChatSource = {
  title?: string
  label?: string
  source?: string
  url?: string
  section?: string
  name?: string
  snippet?: string
  text?: string
  score?: number
  [key: string]: unknown
}

export type ChatResponse = {
  message?: string
  answer: string
  intent?: ChatIntent
  normalized_drugs?: unknown[]
  sources?: ChatSource[]
  warnings?: string[]
  disclaimer?: string
}

export type DrugChatMessageItem = {
  id: string
  role: ChatRole
  content: string
  status?: 'loading' | 'error'
  response?: ChatResponse
}
