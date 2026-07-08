import type { RiskLevel } from './chat'
import type { DoctorMemoryResponse } from './doctorNote'

export type PatientContext = {
  age: number | null
  sex: string
  allergies: string
  pregnancy_status: string
  pregnancy_lactation?: string
  renal_function: string
  hepatic_function: string
  diagnoses: string[]
  current_medications: string
}

export type PrescriptionAuditRequest = {
  prescription_text: string
  patient_context: PatientContext
  use_gemini: boolean
  query_types: string[]
  top_k_per_type: number
}

export type EvidenceItem = {
  chunk_id?: string | null
  slug?: string | null
  section?: string | null
  section_title?: string | null
  source?: string | null
  source_type?: string | null
  url?: string | null
  snippet?: string | null
}

export type MedicationSummary = {
  raw_name?: string | null
  raw_line?: string | null
  instruction?: string | null
  generic_text?: string | null
  brand_text?: string | null
  mapping_status?: string | null
  requires_review?: boolean
  warnings?: string[] | null
  active_ingredients?: Array<{
    name?: string | null
    evidence_slug?: string | null
    strength_raw?: string | null
    strength_value?: number | null
    strength_unit?: string | null
  }> | null
}

export type NormalizedMedication = {
  raw_name?: string | null
  raw_line?: string | null
  generic_text?: string | null
  brand_text?: string | null
  active_ingredients?: Array<{
    name?: string | null
    normalized_name?: string | null
    evidence_slug?: string | null
  }> | null
}

export type PrescriptionCheck = {
  patient_context?: Partial<PatientContext> | null
  normalized_result?: {
    medications?: NormalizedMedication[] | null
    unique_evidence_slugs?: string[] | null
  } | null
}

export type RiskItem = {
  risk_type?: string | null
  severity?: RiskLevel | string | null
  title?: string | null
  explanation?: string | null
  affected_slugs?: string[] | null
  evidence_refs?: string[] | null
  recommendation?: string | null
  evidence?: EvidenceItem[] | null
}

export type Report = {
  status?: string | null
  overall_risk_level?: RiskLevel | string | null
  summary?: string | null
  patient_context?: Partial<PatientContext> | null
  medication_summary?: MedicationSummary[] | null
  medications_requiring_review?: MedicationSummary[] | null
  risk_items?: RiskItem[] | null
  missing_information?: string[] | null
  evidence_sources?: EvidenceItem[] | null
  safety_disclaimer?: string | null
  doctor_facing_response?: string | null
  doctor_facing_warnings?: string[] | null
  doctor_memory?: DoctorMemoryResponse | null
  warnings?: string[] | null
  errors?: string[] | null
}

export type RiskAnalysis = {
  status?: string | null
  overall_risk_level?: RiskLevel | string | null
  risk_items?: RiskItem[] | null
  missing_information?: string[] | null
  warnings?: string[] | null
  errors?: string[] | null
}

export type PrescriptionAuditResponse = {
  status?: string | null
  warnings?: string[] | null
  errors?: string[] | null
  risk_analysis?: RiskAnalysis | null
  prescription_check?: PrescriptionCheck | null
  report?: Report | null
  doctor_memory?: DoctorMemoryResponse | null
}

export type ReportHistoryRead = {
  id: number
  prescription_history_id: number
  report_status?: string | null
  summary?: string | null
  doctor_facing_response?: string | null
  report_payload?: Record<string, unknown> | null
  created_at: string
}

export type PrescriptionHistoryListItem = {
  id: number
  status?: string | null
  overall_risk_level?: string | null
  report_status?: string | null
  created_at: string
}

export type PrescriptionHistoryDetail = {
  id: number
  prescription_text: string
  patient_context?: Record<string, unknown> | null
  query_types?: string[] | null
  use_gemini: boolean
  top_k_per_type: number
  status?: string | null
  overall_risk_level?: string | null
  warnings?: unknown[] | null
  errors?: unknown[] | null
  audit_payload?: Record<string, unknown> | null
  report?: ReportHistoryRead | null
  created_at: string
}
