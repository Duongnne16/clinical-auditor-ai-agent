import type { ReactNode } from 'react'
import type { ChatMessageItem } from '../types/chat'
import type {
  EvidenceItem,
  MedicationSummary,
  PrescriptionAuditResponse,
  RiskItem,
} from '../types/prescriptionAudit'
import RiskBadge from './RiskBadge'

type ChatMessageProps = {
  message: ChatMessageItem
}

const INTERNAL_MAPPING_CODES = new Set([
  'no_mapping_found',
  'drug_mapping_not_found',
  'drug_or_ingredient_not_found',
  'ingredient_evidence_requires_review',
  'safety_mapping_requires_review',
  'safety_unresolved_medications',
  'some_medications_require_review',
])

const MISSING_INFORMATION_LABELS: Record<string, string> = {
  pregnancy_status: 'Tình trạng thai kỳ/cho con bú chưa được ghi nhận.',
  pregnancy_lactation: 'Tình trạng thai kỳ/cho con bú chưa được ghi nhận.',
  hepatic_function: 'Chức năng gan chưa được ghi nhận hoặc cần xác nhận.',
  renal_function: 'Chức năng thận chưa được ghi nhận hoặc cần xác nhận.',
  current_medications: 'Các thuốc bệnh nhân đang dùng ngoài đơn thuốc chưa được ghi nhận.',
  allergies: 'Tiền sử dị ứng thuốc chưa được ghi nhận.',
  diagnoses: 'Bệnh nền/chẩn đoán chưa được ghi nhận đầy đủ.',
}

const EVIDENCE_LABELS: Record<string, string> = {
  trungtamthuoc: 'Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam',
  levofloxacin: 'Levofloxacin',
  omeprazole: 'Omeprazole',
  sucralfate: 'Sucralfate',
  tuong_tac_thuoc: 'Tương tác thuốc',
  than_trong: 'Thận trọng',
  chong_chi_dinh: 'Chống chỉ định',
  thai_ky_cho_con_bu: 'Thai kỳ/cho con bú',
  lieu_luong_va_cach_dung: 'Liều lượng và cách dùng',
  tac_dung_khong_mong_muon: 'Tác dụng không mong muốn',
}

const asTextArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter(Boolean).map(String) : []

const dedupe = (values: string[]): string[] => Array.from(new Set(values))

const hasItems = <T,>(value: T[] | null | undefined): value is T[] =>
  Array.isArray(value) && value.length > 0

const medicationName = (medication: MedicationSummary): string =>
  medication.raw_line ||
  medication.raw_name ||
  medication.generic_text ||
  medication.brand_text ||
  'Không rõ tên thuốc'

const activeIngredientText = (medication: MedicationSummary): string => {
  if (!Array.isArray(medication.active_ingredients)) {
    return ''
  }

  return medication.active_ingredients
    .map((ingredient) => ingredient.name)
    .filter(Boolean)
    .join(', ')
}

const getRiskItems = (auditResult: PrescriptionAuditResponse): RiskItem[] => {
  if (hasItems(auditResult.report?.risk_items)) {
    return auditResult.report.risk_items
  }

  if (hasItems(auditResult.risk_analysis?.risk_items)) {
    return auditResult.risk_analysis.risk_items
  }

  return []
}

const getMergedWarnings = (auditResult: PrescriptionAuditResponse): string[] =>
  dedupe([
    ...asTextArray(auditResult.warnings),
    ...asTextArray(auditResult.report?.warnings),
    ...asTextArray(auditResult.risk_analysis?.warnings),
  ])

const getMergedErrors = (auditResult: PrescriptionAuditResponse): string[] =>
  dedupe([
    ...asTextArray(auditResult.errors),
    ...asTextArray(auditResult.report?.errors),
    ...asTextArray(auditResult.risk_analysis?.errors),
  ])

const getDoctorWarnings = (auditResult: PrescriptionAuditResponse): string[] => {
  const explicit = asTextArray(auditResult.report?.doctor_facing_warnings)
  const technical = getMergedWarnings(auditResult)
  const derived = technical.some((warning) => INTERNAL_MAPPING_CODES.has(warning))
    ? ['Một số dòng thuốc chưa được hệ thống nhận diện chắc chắn, cần rà soát lại.']
    : []

  return dedupe([...explicit, ...derived])
}

const readableLabel = (value?: string | null): string => {
  const text = String(value || '').trim()
  if (!text) {
    return ''
  }
  return EVIDENCE_LABELS[text.toLowerCase()] || text
}

const evidenceLabel = (evidence: EvidenceItem): string => {
  const labels = [
    readableLabel(evidence.source || evidence.source_type || 'Nguồn tham khảo'),
    readableLabel(evidence.slug),
    readableLabel(evidence.section_title || evidence.section),
  ].filter(Boolean)

  return dedupe(labels).join(' — ') || 'Nguồn tham khảo'
}

const uniqueEvidence = (items: EvidenceItem[] | null | undefined): EvidenceItem[] => {
  if (!Array.isArray(items)) {
    return []
  }

  const seen = new Set<string>()
  const output: EvidenceItem[] = []
  for (const item of items) {
    const key = item.url || evidenceLabel(item)
    if (!key || seen.has(key)) {
      continue
    }
    seen.add(key)
    output.push(item)
  }
  return output
}

const collectSourceReferences = (riskItems: RiskItem[]): EvidenceItem[] => {
  const seen = new Set<string>()
  const output: EvidenceItem[] = []

  for (const item of riskItems) {
    if (!Array.isArray(item.evidence)) {
      continue
    }

    for (const evidence of item.evidence) {
      const label = evidenceLabel(evidence)
      const key = evidence.url || evidence.chunk_id || label
      if (!label || !key || seen.has(key)) {
        continue
      }
      seen.add(key)
      output.push(evidence)
    }
  }

  return output
}

const mapMissingInformation = (value: string[] | null | undefined): string[] =>
  dedupe(
    asTextArray(value).map((item) => MISSING_INFORMATION_LABELS[item] || item),
  )

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-5">
      <h3 className="text-sm font-semibold text-gray-950">{title}</h3>
      <div className="mt-2">{children}</div>
    </section>
  )
}

function CollapsedDetails({
  title,
  children,
  className = '',
}: {
  title: string
  children: ReactNode
  className?: string
}) {
  return (
    <details
      className={`mt-5 rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 ${className}`}
    >
      <summary className="cursor-pointer font-medium text-gray-800">{title}</summary>
      <div className="mt-3">{children}</div>
    </details>
  )
}

function RiskCards({ riskItems }: { riskItems: RiskItem[] }) {
  if (!hasItems(riskItems)) {
    return null
  }

  return (
    <div className="space-y-3">
      {riskItems.map((item, index) => {
        const evidenceItems = uniqueEvidence(item.evidence)

        return (
          <article
            key={`${item.title || 'risk'}-${index}`}
            className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm"
          >
            <div className="flex flex-wrap items-start gap-2">
              <RiskBadge level={item.severity} />
              <h4 className="min-w-0 flex-1 text-sm font-semibold text-gray-950">
                {item.title || 'Điểm cần lưu ý'}
              </h4>
            </div>
            {item.explanation ? (
              <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-gray-700">
                <span className="font-semibold text-gray-900">
                  Nội dung đánh giá:{' '}
                </span>
                {item.explanation}
              </p>
            ) : null}
            {item.recommendation ? (
              <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-gray-700">
                <span className="font-semibold text-gray-900">
                  Gợi ý rà soát:{' '}
                </span>
                {item.recommendation}
              </p>
            ) : null}
            {hasItems(evidenceItems) ? (
              <div className="mt-3">
                <p className="text-xs font-semibold text-gray-700">
                  Nguồn tham khảo
                </p>
                <ul className="mt-1 space-y-1 text-xs leading-5 text-gray-600">
                  {evidenceItems.map((evidence) => (
                    <li key={evidence.url || evidenceLabel(evidence)}>
                      <span>{evidenceLabel(evidence)}</span>
                      {evidence.url ? (
                        <a
                          href={evidence.url}
                          target="_blank"
                          rel="noreferrer"
                          className="ml-2 font-medium text-blue-700 underline-offset-2 hover:underline"
                        >
                          Xem nguồn
                        </a>
                      ) : null}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </article>
        )
      })}
    </div>
  )
}

function MedicationsReview({
  medications,
}: {
  medications: MedicationSummary[]
}) {
  if (!hasItems(medications)) {
    return null
  }

  return (
    <div className="space-y-2">
      {medications.map((medication, index) => {
        const ingredients = activeIngredientText(medication)

        return (
          <div
            key={`${medicationName(medication)}-${index}`}
            className="rounded-xl border border-gray-200 bg-white px-4 py-3"
          >
            <p className="text-sm font-medium text-gray-900">
              {medicationName(medication)}
            </p>
            {medication.instruction ? (
              <p className="mt-1 text-xs text-gray-600">
                Hướng dẫn dùng: {medication.instruction}
              </p>
            ) : null}
            {ingredients ? (
              <p className="mt-1 text-xs text-gray-500">
                Hoạt chất: {ingredients}
              </p>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}

function MissingInformationList({ items }: { items: string[] }) {
  if (!hasItems(items)) {
    return null
  }

  return (
    <ul className="list-disc space-y-1 pl-5 text-sm text-gray-700">
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  )
}

function TechnicalDetails({
  warnings,
  errors,
  riskItems,
}: {
  warnings: string[]
  errors: string[]
  riskItems: RiskItem[]
}) {
  const chunkIds = dedupe(
    riskItems.flatMap((item) => [
      ...asTextArray(item.evidence_refs),
      ...(item.evidence || [])
        .map((evidence) => evidence.chunk_id || '')
        .filter(Boolean),
    ]),
  )

  if (!hasItems(warnings) && !hasItems(errors) && !hasItems(chunkIds)) {
    return null
  }

  return (
    <CollapsedDetails title="Chi tiết kỹ thuật" className="text-xs text-gray-600">
      {hasItems(warnings) ? (
        <div>
          <p className="font-medium">Warnings</p>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {hasItems(errors) ? (
        <div className="mt-3">
          <p className="font-medium">Errors</p>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-red-700">
            {errors.map((error) => (
              <li key={error}>{error}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {hasItems(chunkIds) ? (
        <div className="mt-3">
          <p className="font-medium">Evidence chunk IDs</p>
          <p className="mt-1 break-words">{chunkIds.join(', ')}</p>
        </div>
      ) : null}
    </CollapsedDetails>
  )
}

function ClinicalDetails({
  riskItems,
  medicationsRequiringReview,
  missingInformation,
}: {
  riskItems: RiskItem[]
  medicationsRequiringReview: MedicationSummary[]
  missingInformation: string[]
}) {
  const hasClinicalDetails =
    hasItems(riskItems) ||
    hasItems(medicationsRequiringReview) ||
    hasItems(missingInformation)

  if (!hasClinicalDetails) {
    return null
  }

  return (
    <CollapsedDetails title="Chi tiết đánh giá">
      {hasItems(riskItems) ? (
        <Section title="Các điểm cần bác sĩ/dược sĩ rà soát">
          <RiskCards riskItems={riskItems} />
        </Section>
      ) : null}

      {hasItems(medicationsRequiringReview) ? (
        <Section title="Thuốc/dòng cần rà soát lại">
          <MedicationsReview medications={medicationsRequiringReview} />
        </Section>
      ) : null}

      {hasItems(missingInformation) ? (
        <Section title="Thông tin cần xác nhận">
          <MissingInformationList items={missingInformation} />
        </Section>
      ) : null}
    </CollapsedDetails>
  )
}

function SourceReferences({ riskItems }: { riskItems: RiskItem[] }) {
  const sources = collectSourceReferences(riskItems)

  if (!hasItems(sources)) {
    return null
  }

  return (
    <section className="mt-5 rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 text-xs leading-5 text-gray-600">
      <h3 className="font-semibold text-gray-800">Nguồn tham khảo</h3>
      <ol className="mt-2 space-y-1">
        {sources.map((source, index) => (
          <li key={source.url || source.chunk_id || evidenceLabel(source)}>
            <span className="font-medium text-gray-500">[{index + 1}]</span>{' '}
            <span>{evidenceLabel(source)}</span>
            {source.url ? (
              <>
                <span className="text-gray-400"> · </span>
                <a
                  href={source.url}
                  target="_blank"
                  rel="noreferrer"
                  className="font-medium text-blue-700 underline-offset-2 hover:underline"
                >
                  Xem nguồn
                </a>
              </>
            ) : null}
          </li>
        ))}
      </ol>
    </section>
  )
}

function StructuredFallbackAuditResult({
  result,
}: {
  result: PrescriptionAuditResponse
}) {
  const report = result.report
  const riskItems = getRiskItems(result)
  const warnings = getMergedWarnings(result)
  const errors = getMergedErrors(result)
  const doctorWarnings = getDoctorWarnings(result)
  const overallRisk =
    report?.overall_risk_level ||
    result.risk_analysis?.overall_risk_level ||
    'unknown'
  const missingInformation = asTextArray(report?.missing_information)
  const medicationsRequiringReview = Array.isArray(
    report?.medications_requiring_review,
  )
    ? report.medications_requiring_review
    : []

  return (
    <div>
      <h2 className="text-base font-semibold text-gray-950">
        Kết quả kiểm tra đơn thuốc
      </h2>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-gray-700">
          Mức ưu tiên rà soát
        </span>
        <RiskBadge level={overallRisk} />
      </div>

      {report?.summary ? (
        <p className="mt-4 whitespace-pre-wrap text-sm leading-7 text-gray-900">
          {report.summary}
        </p>
      ) : (
        <p className="mt-4 text-sm leading-7 text-gray-600">
          API đã trả về phản hồi nhưng chưa có báo cáo chi tiết.
        </p>
      )}

      {hasItems(doctorWarnings) ? (
        <Section title="Điểm cần lưu ý">
          <MissingInformationList items={doctorWarnings} />
        </Section>
      ) : null}

      {hasItems(riskItems) ? (
        <Section title="Các điểm cần bác sĩ/dược sĩ rà soát">
          <RiskCards riskItems={riskItems} />
        </Section>
      ) : null}

      {hasItems(medicationsRequiringReview) ? (
        <Section title="Thuốc/dòng cần rà soát lại">
          <MedicationsReview medications={medicationsRequiringReview} />
        </Section>
      ) : null}

      {hasItems(missingInformation) ? (
        <Section title="Thông tin cần bổ sung">
          <MissingInformationList items={missingInformation} />
        </Section>
      ) : null}

      {report?.safety_disclaimer ? (
        <p className="mt-5 rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 text-xs leading-5 text-gray-600">
          {report.safety_disclaimer}
        </p>
      ) : null}

      <TechnicalDetails warnings={warnings} errors={errors} riskItems={riskItems} />
    </div>
  )
}

function DoctorFacingAuditResult({
  result,
  doctorFacingResponse,
}: {
  result: PrescriptionAuditResponse
  doctorFacingResponse: string
}) {
  const report = result.report
  const riskItems = getRiskItems(result)
  const warnings = getMergedWarnings(result)
  const errors = getMergedErrors(result)
  const missingInformation = mapMissingInformation(report?.missing_information)
  const medicationsRequiringReview = Array.isArray(
    report?.medications_requiring_review,
  )
    ? report.medications_requiring_review
    : []
  const showDebugDetails = import.meta.env.VITE_SHOW_DEBUG_DETAILS === 'true'

  return (
    <div>
      <div className="whitespace-pre-wrap text-sm leading-7 text-gray-900">
        {doctorFacingResponse}
      </div>

      <SourceReferences riskItems={riskItems} />

      {showDebugDetails ? (
        <>
          <ClinicalDetails
            riskItems={riskItems}
            medicationsRequiringReview={medicationsRequiringReview}
            missingInformation={missingInformation}
          />
          <TechnicalDetails warnings={warnings} errors={errors} riskItems={riskItems} />
        </>
      ) : null}
    </div>
  )
}

function AuditResult({ result }: { result: PrescriptionAuditResponse }) {
  const doctorFacingResponse = result.report?.doctor_facing_response?.trim()

  if (doctorFacingResponse) {
    return (
      <DoctorFacingAuditResult
        result={result}
        doctorFacingResponse={doctorFacingResponse}
      />
    )
  }

  return <StructuredFallbackAuditResult result={result} />
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end px-4 py-3 sm:px-6">
        <div className="max-w-[min(42rem,85%)] rounded-3xl bg-gray-100 px-5 py-3 text-sm leading-6 text-gray-900 shadow-sm whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="px-4 py-5 sm:px-6">
      <div className="mx-auto flex w-full max-w-3xl gap-4">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-emerald-200 bg-emerald-50 text-sm font-semibold text-emerald-700">
          AI
        </div>
        <div className="min-w-0 flex-1 text-sm leading-7 text-gray-900">
          {message.auditResult ? (
            <AuditResult result={message.auditResult} />
          ) : (
            <div
              className={`whitespace-pre-wrap ${
                message.status === 'error' ? 'text-red-700' : ''
              }`}
            >
              {message.status === 'loading' ? (
                <span className="inline-flex items-center gap-2">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
                  {message.content}
                </span>
              ) : (
                message.content
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
