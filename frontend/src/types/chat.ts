import type { PrescriptionAuditResponse } from './prescriptionAudit'

export type ChatRole = 'user' | 'assistant'

export type RiskLevel = 'high' | 'moderate' | 'low' | 'unknown'

export type ChatMessageItem = {
  id: string
  role: ChatRole
  content: string
  status?: 'loading' | 'error'
  auditResult?: PrescriptionAuditResponse
}
