export type RiskLevel = 'high' | 'moderate' | 'low' | 'unknown'

export type ChatRole = 'user' | 'assistant'

export type ChatMessageItem = {
  id: string
  role: ChatRole
  content: string
  riskLevel?: RiskLevel
}
