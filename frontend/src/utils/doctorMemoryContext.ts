import type { DoctorNoteCreatePayload } from '../types/doctorNote'
import type {
  NormalizedMedication,
  PatientContext,
  PrescriptionAuditResponse,
} from '../types/prescriptionAudit'

const STRENGTH_OR_QUANTITY_PATTERN =
  /\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|kg|ml|iu|ui|%)\b|\b\d+\s*(?:viên|vien|gói|goi|ống|ong|lọ|lo|lần|lan|ngày|ngay)\b/gi

const ROUTE_WORDS = new Set([
  'ngay',
  'uống',
  'uong',
  'lần',
  'lan',
  'mỗi',
  'moi',
  'viên',
  'vien',
  'gói',
  'goi',
  'sáng',
  'sang',
  'chiều',
  'chieu',
  'tối',
  'toi',
  'trưa',
  'trua',
])

const normalizeText = (value: unknown): string =>
  String(value || '').trim().replace(/\s+/g, ' ')

const foldText = (value: unknown): string =>
  normalizeText(value)
    .toLowerCase()
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .replace(/\u0111/g, 'd')
    .replace(/đ/g, 'd')

const dedupe = (values: string[]): string[] => Array.from(new Set(values))

const cleanDrugName = (value: unknown): string => {
  const folded = foldText(value)
    .replace(/\([^)]*\)/g, ' ')
    .replace(STRENGTH_OR_QUANTITY_PATTERN, ' ')
    .replace(/[^a-z0-9\s-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  return folded
}

const simpleDrugNameFromText = (value: unknown): string => {
  const cleaned = cleanDrugName(value)
  const tokens = cleaned
    .split(/\s+/)
    .filter((token) => token.length >= 3 && !ROUTE_WORDS.has(token))

  return tokens[0] || ''
}

const medicationFallbackName = (medication: NormalizedMedication): string => {
  const candidates = [
    medication.generic_text,
    medication.raw_line,
    medication.raw_name,
    medication.brand_text,
  ]

  for (const candidate of candidates) {
    const name = simpleDrugNameFromText(candidate)
    if (name) {
      return name
    }
  }

  return ''
}

export const extractDoctorMemoryDrugNames = (
  auditResult: PrescriptionAuditResponse | null,
): string[] => {
  const medications =
    auditResult?.prescription_check?.normalized_result?.medications || []

  const names: string[] = []
  for (const medication of medications) {
    const ingredients = Array.isArray(medication.active_ingredients)
      ? medication.active_ingredients
      : []

    const ingredientNames = ingredients
      .map(
        (ingredient) =>
          ingredient.evidence_slug || ingredient.normalized_name || ingredient.name,
      )
      .map(cleanDrugName)
      .filter(Boolean)

    if (ingredientNames.length > 0) {
      names.push(...ingredientNames)
      continue
    }

    const fallback = medicationFallbackName(medication)
    if (fallback) {
      names.push(fallback)
    }
  }

  return dedupe(names)
}

export const buildDrugPairKeys = (drugNames: string[]): string[] => {
  const sorted = [...dedupe(drugNames)].sort()
  const pairs: string[] = []

  for (let leftIndex = 0; leftIndex < sorted.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < sorted.length; rightIndex += 1) {
      pairs.push(`${sorted[leftIndex]}|${sorted[rightIndex]}`)
    }
  }

  return pairs
}

const contextFromAudit = (
  auditResult: PrescriptionAuditResponse | null,
): Partial<PatientContext> => {
  return (
    auditResult?.prescription_check?.patient_context ||
    auditResult?.report?.patient_context ||
    {}
  )
}

const extractDiagnosisKeywords = (
  auditResult: PrescriptionAuditResponse | null,
): string[] => {
  const context = contextFromAudit(auditResult)
  const diagnoses = Array.isArray(context.diagnoses) ? context.diagnoses : []

  return dedupe(diagnoses.map(foldText).filter(Boolean))
}

const inferPatientTags = (
  auditResult: PrescriptionAuditResponse | null,
): string[] => {
  const context = contextFromAudit(auditResult)
  const tags: string[] = []
  const pregnancy = foldText(
    context.pregnancy_lactation || context.pregnancy_status,
  )

  if (pregnancy.includes('mang thai') || pregnancy.includes('pregnant')) {
    tags.push('pregnancy')
  } else if (pregnancy.includes('khong') || pregnancy.includes('no')) {
    tags.push('not_pregnant')
  }

  const renal = foldText(context.renal_function)
  if (
    renal.includes('suy than') ||
    renal.includes('egfr thap') ||
    renal.includes('giam') ||
    renal.includes('renal impairment')
  ) {
    tags.push('renal_impairment')
  }

  const hepatic = foldText(context.hepatic_function)
  if (
    hepatic.includes('suy gan') ||
    hepatic.includes('tang men gan') ||
    hepatic.includes('giam') ||
    hepatic.includes('hepatic impairment')
  ) {
    tags.push('hepatic_impairment')
  }

  const diagnosisText = extractDiagnosisKeywords(auditResult).join(' ')
  if (diagnosisText.includes('soi than')) {
    tags.push('renal_stone')
  }

  return dedupe(tags)
}

const displayDrugName = (value: string): string =>
  value
    .split(/[\s-]+/)
    .map((part) => (part ? `${part[0].toUpperCase()}${part.slice(1)}` : part))
    .join(' ')

const inferTitle = (drugNames: string[]): string => {
  if (drugNames.length > 1) {
    return 'Ghi chú đơn thuốc'
  }

  if (drugNames.length === 1) {
    return `Ghi chú về ${displayDrugName(drugNames[0])}`
  }

  return 'Ghi chú đơn thuốc'
}

export const buildDoctorNotePayload = (
  auditResult: PrescriptionAuditResponse,
  noteText: string,
): DoctorNoteCreatePayload => {
  const drugNames = extractDoctorMemoryDrugNames(auditResult)

  return {
    content: noteText,
    note_text: noteText,
    title: inferTitle(drugNames),
    note_type: 'clinical_experience',
    source_context: 'prescription_audit',
    active_ingredients: drugNames,
    drug_pair_keys: buildDrugPairKeys(drugNames),
    diagnosis_keywords: extractDiagnosisKeywords(auditResult),
    patient_tags: inferPatientTags(auditResult),
    applicability: {},
    priority: 'normal',
  }
}
