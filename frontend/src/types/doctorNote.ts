export type DoctorMemoryNote = {
  note_id?: string
  id?: string | number
  title?: string | null
  note_text?: string | null
  content?: string | null
  note_type?: string | null
  source_context?: string | null
  active_ingredients?: string[] | null
  drug_pair_keys?: string[] | null
  diagnosis_keywords?: string[] | null
  patient_tags?: string[] | null
  applicability?: Record<string, unknown> | null
  priority?: string | null
  score?: number | null
  match_reason?: string | null
  created_at?: string | null
  is_newly_saved?: boolean
}

export type DoctorMemoryResponse = {
  matched_notes: DoctorMemoryNote[]
  warnings?: string[] | null
}

export type DoctorNoteCreatePayload = {
  content: string
  note_text: string
  title?: string
  note_type?: string
  source_context: 'prescription_audit'
  active_ingredients: string[]
  drug_pair_keys: string[]
  diagnosis_keywords: string[]
  patient_tags: string[]
  applicability: Record<string, unknown>
  priority: 'normal' | 'high'
}
