import type { ChatRole } from './chat'
import type { PrescriptionAuditResponse } from './prescriptionAudit'

export type AuditMessageItem = {
  id: string
  role: ChatRole
  content: string
  status?: 'loading' | 'error'
  auditResult?: PrescriptionAuditResponse
}
