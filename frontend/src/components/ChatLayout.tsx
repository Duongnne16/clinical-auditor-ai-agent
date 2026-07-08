import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Clock3,
  FileText,
  History,
  Loader2,
  LogOut,
  PanelRightOpen,
  Plus,
} from 'lucide-react'
import {
  auditPrescription,
  getPrescriptionHistory,
  listPrescriptionHistory,
} from '../api/prescriptionAuditApi'
import ChatInput from './ChatInput'
import ChatMessage from './ChatMessage'
import DrugChatPanel from './DrugChatPanel'
import DoctorMemoryPanel from './DoctorMemoryPanel'
import type { AuditMessageItem } from '../types/auditConversation'
import type { DoctorMemoryNote } from '../types/doctorNote'
import type {
  PatientContext,
  PrescriptionAuditRequest,
  PrescriptionAuditResponse,
  PrescriptionHistoryDetail,
  PrescriptionHistoryListItem,
} from '../types/prescriptionAudit'

type DemoCase = {
  id: string
  label: string
  prescriptionText: string
  patientContext: PatientContext
  queryTypes: string[]
}

type CenterMode = 'audit' | 'drug_chat'

type ChatLayoutProps = {
  onLogout: () => void
}

const demoCases: DemoCase[] = [
  {
    id: 'metformin-egfr',
    label: 'Metformin + eGFR 25',
    prescriptionText: `Metformin (Panfor SR) 750mg x 30 viên
Omeprazol (Kagascdine) 20mg x 14 viên
Paracetamol (Hapacol) 500mg x 10 viên`,
    patientContext: {
      age: 60,
      sex: 'male',
      allergies: 'none reported',
      pregnancy_status: 'not_applicable',
      renal_function: 'eGFR 25 ml/min/1.73m2',
      hepatic_function: 'no known hepatic impairment',
      diagnoses: ['type 2 diabetes', 'gastritis'],
      current_medications: 'none reported',
    },
    queryTypes: ['contraindication', 'renal_hepatic', 'interaction'],
  },
  {
    id: 'clopidogrel-omeprazole',
    label: 'Clopidogrel + Omeprazole',
    prescriptionText: `Clopidogrel (Plavix) 75mg x 30 viên
Omeprazol (Kagascdine) 20mg x 14 viên
Atorvastatin 20mg x 30 viên`,
    patientContext: {
      age: 68,
      sex: 'male',
      allergies: 'none reported',
      pregnancy_status: 'not_applicable',
      renal_function: 'eGFR 70 ml/min/1.73m2',
      hepatic_function: 'no known hepatic impairment',
      diagnoses: ['coronary artery disease', 'gastritis', 'dyslipidemia'],
      current_medications: 'none reported',
    },
    queryTypes: ['interaction', 'contraindication', 'precaution'],
  },
]

const defaultPatientContext: PatientContext = {
  age: null,
  sex: 'unknown',
  allergies: 'not provided',
  pregnancy_status: 'unknown',
  renal_function: 'not provided',
  hepatic_function: 'not provided',
  diagnoses: [],
  current_medications: 'not provided',
}

const createMessageId = () => crypto.randomUUID()
const HISTORY_PAGE_SIZE = 20

const isObjectRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)

const mergeHistoryItems = (
  currentItems: PrescriptionHistoryListItem[],
  nextItems: PrescriptionHistoryListItem[],
): PrescriptionHistoryListItem[] => {
  const seenIds = new Set<number>()
  const merged: PrescriptionHistoryListItem[] = []

  for (const item of [...currentItems, ...nextItems]) {
    if (seenIds.has(item.id)) {
      continue
    }
    seenIds.add(item.id)
    merged.push(item)
  }

  return merged
}

const formatHistoryDate = (value: string): string => {
  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return 'Không rõ thời gian'
  }

  return new Intl.DateTimeFormat('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

const historyStatusText = (item: PrescriptionHistoryListItem): string => {
  const parts = [item.overall_risk_level, item.status]
    .map((value) => String(value || '').trim())
    .filter(Boolean)

  return parts.length > 0 ? parts.join(' Â· ') : 'Chưa có trạng thái'
}

const buildFallbackAuditResponse = (
  detail: PrescriptionHistoryDetail,
): PrescriptionAuditResponse => {
  const reportPayload = isObjectRecord(detail.report?.report_payload)
    ? detail.report.report_payload
    : null
  const report = reportPayload
    ? (reportPayload as PrescriptionAuditResponse['report'])
    : detail.report
      ? {
          status: detail.report.report_status,
          summary: detail.report.summary,
          doctor_facing_response: detail.report.doctor_facing_response,
        }
      : null

  return {
    status: detail.status,
    warnings: Array.isArray(detail.warnings) ? detail.warnings.map(String) : [],
    errors: Array.isArray(detail.errors) ? detail.errors.map(String) : [],
    report,
  }
}

function HistoryListItem({
  item,
  isLoading,
  disabled,
  onClick,
}: {
  item: PrescriptionHistoryListItem
  isLoading: boolean
  disabled: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="flex w-full items-start gap-2 rounded-lg px-3 py-2 text-left text-xs text-gray-700 transition hover:bg-gray-100 hover:text-gray-950 focus:outline-none focus:ring-2 focus:ring-gray-300 disabled:cursor-wait disabled:opacity-60"
    >
      {isLoading ? (
        <Loader2 className="mt-0.5 shrink-0 animate-spin" size={15} />
      ) : (
        <Clock3 className="mt-0.5 shrink-0" size={15} />
      )}
      <span className="min-w-0 flex-1">
        <span className="block font-medium text-gray-900">
          {formatHistoryDate(item.created_at)}
        </span>
        <span className="mt-0.5 block truncate text-gray-600">
          {historyStatusText(item)}
        </span>
        {item.report_status ? (
          <span className="mt-0.5 block truncate text-gray-500">
            Report: {item.report_status}
          </span>
        ) : null}
      </span>
    </button>
  )
}

export default function ChatLayout({ onLogout }: ChatLayoutProps) {
  const [centerMode, setCenterMode] = useState<CenterMode>('audit')
  const [messages, setMessages] = useState<AuditMessageItem[]>([])
  const [inputValue, setInputValue] = useState('')
  const [selectedDemoId, setSelectedDemoId] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [inputFocusSignal, setInputFocusSignal] = useState(0)
  const [latestAuditResult, setLatestAuditResult] =
    useState<PrescriptionAuditResponse | null>(null)
  const [relatedMemoryNotes, setRelatedMemoryNotes] = useState<
    DoctorMemoryNote[]
  >([])
  const [isMemoryPanelOpen, setIsMemoryPanelOpen] = useState(true)
  const [isMemoryDrawerOpen, setIsMemoryDrawerOpen] = useState(false)
  const [historyItems, setHistoryItems] = useState<PrescriptionHistoryListItem[]>(
    [],
  )
  const [historyOffset, setHistoryOffset] = useState(0)
  const [hasMoreHistory, setHasMoreHistory] = useState(false)
  const [isHistoryLoading, setIsHistoryLoading] = useState(false)
  const [isHistoryLoadingMore, setIsHistoryLoadingMore] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [loadingHistoryId, setLoadingHistoryId] = useState<number | null>(null)
  const messageEndRef = useRef<HTMLDivElement | null>(null)
  const messageRefs = useRef<Record<string, HTMLDivElement | null>>({})

  const loadHistoryPage = useCallback(
    async ({ offset, append }: { offset: number; append: boolean }) => {
      if (append) {
        setIsHistoryLoadingMore(true)
      } else {
        setIsHistoryLoading(true)
      }
      setHistoryError('')

      try {
        const nextItems = await listPrescriptionHistory({
          limit: HISTORY_PAGE_SIZE,
          offset,
        })
        setHistoryItems((currentItems) =>
          append ? mergeHistoryItems(currentItems, nextItems) : nextItems,
        )
        setHistoryOffset(offset + nextItems.length)
        setHasMoreHistory(nextItems.length === HISTORY_PAGE_SIZE)
      } catch (error) {
        setHistoryError(
          error instanceof Error
            ? error.message
            : 'Không thể tải lịch sử kiểm tra.',
        )
      } finally {
        setIsHistoryLoading(false)
        setIsHistoryLoadingMore(false)
      }
    },
    [],
  )

  const refreshHistory = useCallback(async () => {
    await loadHistoryPage({ offset: 0, append: false })
  }, [loadHistoryPage])

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      void refreshHistory()
    }, 0)

    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [refreshHistory])

  useEffect(() => {
    const latestMessage = messages[messages.length - 1]
    const hasDoctorFacingResponse =
      latestMessage?.role === 'assistant' &&
      Boolean(latestMessage.auditResult?.report?.doctor_facing_response?.trim())

    if (hasDoctorFacingResponse) {
      messageRefs.current[latestMessage.id]?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      })
      return
    }

    messageEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const selectedDemo = demoCases.find((demo) => demo.id === selectedDemoId)

  const handleInputChange = (nextValue: string) => {
    setInputValue(nextValue)

    if (selectedDemo && nextValue !== selectedDemo.prescriptionText) {
      setSelectedDemoId(null)
    }
  }

  const handleDemoClick = (demo: DemoCase) => {
    setCenterMode('audit')
    setInputValue(demo.prescriptionText)
    setSelectedDemoId(demo.id)
    setInputFocusSignal((currentSignal) => currentSignal + 1)
  }

  const buildAuditRequest = (
    prescriptionText: string,
  ): PrescriptionAuditRequest => {
    const activeDemo =
      selectedDemo && prescriptionText === selectedDemo.prescriptionText
        ? selectedDemo
        : null

    return {
      prescription_text: prescriptionText,
      patient_context: activeDemo
        ? activeDemo.patientContext
        : defaultPatientContext,
      use_gemini: true,
      query_types: activeDemo
        ? activeDemo.queryTypes
        : ['interaction', 'contraindication', 'precaution'],
      top_k_per_type: 5,
    }
  }

  const replaceAssistantMessage = (
    messageId: string,
    nextMessage: Partial<AuditMessageItem>,
  ) => {
    setMessages((currentMessages) =>
      currentMessages.map((message) =>
        message.id === messageId ? { ...message, ...nextMessage } : message,
      ),
    )
  }

  const handleHistoryClick = async (historyId: number) => {
    if (loadingHistoryId !== null) {
      return
    }

    setLoadingHistoryId(historyId)
    setHistoryError('')

    try {
      const detail = await getPrescriptionHistory(historyId)
      const auditResult = isObjectRecord(detail.audit_payload)
        ? (detail.audit_payload as PrescriptionAuditResponse)
        : buildFallbackAuditResponse(detail)

      setCenterMode('audit')
      setMessages([
        {
          id: createMessageId(),
          role: 'user',
          content: detail.prescription_text,
        },
        {
          id: createMessageId(),
          role: 'assistant',
          content: '',
          auditResult,
        },
      ])
      setInputValue('')
      setSelectedDemoId(null)
      setLatestAuditResult(auditResult)
      setRelatedMemoryNotes(auditResult.doctor_memory?.matched_notes || [])
      setIsMemoryPanelOpen(true)
      setIsMemoryDrawerOpen(false)
    } catch (error) {
      setHistoryError(
        error instanceof Error
          ? error.message
          : 'Không thể mở lịch sử kiểm tra.',
      )
    } finally {
      setLoadingHistoryId(null)
    }
  }

  const handleSend = async () => {
    const trimmedInput = inputValue.trim()

    if (!trimmedInput || isSubmitting) {
      return
    }

    const loadingMessageId = createMessageId()
    const userMessage: AuditMessageItem = {
      id: createMessageId(),
      role: 'user',
      content: trimmedInput,
    }
    const loadingMessage: AuditMessageItem = {
      id: loadingMessageId,
      role: 'assistant',
      content: 'Đang kiểm tra đơn thuốc...',
      status: 'loading',
    }

    setMessages((currentMessages) => [
      ...currentMessages,
      userMessage,
      loadingMessage,
    ])
    setInputValue('')
    setSelectedDemoId(null)
    setIsSubmitting(true)

    try {
      const auditResult = await auditPrescription(buildAuditRequest(trimmedInput))
      setLatestAuditResult(auditResult)
      setRelatedMemoryNotes(auditResult.doctor_memory?.matched_notes || [])
      setIsMemoryPanelOpen(true)

      replaceAssistantMessage(loadingMessageId, {
        content: '',
        status: undefined,
        auditResult,
      })
      void refreshHistory()
    } catch (error) {
      replaceAssistantMessage(loadingMessageId, {
        content:
          error instanceof Error
            ? error.message
            : 'Không thể kiểm tra đơn thuốc do lỗi không xác định.',
        status: 'error',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleNewAudit = () => {
    setCenterMode('audit')
    setMessages([])
    setInputValue('')
    setSelectedDemoId(null)
    setLatestAuditResult(null)
    setRelatedMemoryNotes([])
    setIsMemoryPanelOpen(true)
    setIsMemoryDrawerOpen(false)
  }

  const handleAddLocalMemoryNote = (note: DoctorMemoryNote) => {
    setRelatedMemoryNotes((currentNotes) => [note, ...currentNotes])
  }

  return (
    <div className="flex h-screen overflow-hidden bg-white text-gray-900">
      <aside className="hidden w-[260px] shrink-0 flex-col border-r border-gray-200 bg-[#f9f9f9] p-3 md:flex">
        <div className="flex h-full flex-col">
          <div className="px-2 py-3">
            <h1 className="text-sm font-semibold text-gray-900">
              Clinical Auditor
            </h1>
            <p className="mt-1 text-xs text-gray-500">AI Agent</p>
          </div>

          <button
            type="button"
            onClick={handleNewAudit}
            className="mt-2 flex w-full items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2.5 text-left text-sm font-medium text-gray-900 transition hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-300"
          >
            <Plus size={16} />
            New audit
          </button>

          <div className="mt-6 space-y-2">
            <div className="flex items-center justify-between gap-2 px-2">
              <p className="text-xs font-medium uppercase tracking-wide text-gray-500">
                Lịch sử kiểm tra
              </p>
              {isHistoryLoading && historyItems.length > 0 ? (
                <Loader2 className="animate-spin text-gray-400" size={14} />
              ) : null}
            </div>

            {isHistoryLoading && historyItems.length === 0 ? (
              <div className="flex items-center gap-2 px-3 py-2 text-xs text-gray-500">
                <Loader2 className="animate-spin" size={14} />
                Đang tải lịch sử...
              </div>
            ) : null}

            {!isHistoryLoading && historyItems.length === 0 && !historyError ? (
              <p className="px-3 py-2 text-xs leading-5 text-gray-500">
                Chua co lan kiem tra nao.
              </p>
            ) : null}

            {historyItems.length > 0 ? (
              <div className="max-h-64 space-y-1 overflow-y-auto pr-1">
                {historyItems.map((item) => (
                  <HistoryListItem
                    key={item.id}
                    item={item}
                    isLoading={loadingHistoryId === item.id}
                    disabled={loadingHistoryId !== null}
                    onClick={() => void handleHistoryClick(item.id)}
                  />
                ))}
              </div>
            ) : null}

            {historyError ? (
              <p className="rounded-lg border border-red-100 bg-red-50 px-3 py-2 text-xs leading-5 text-red-700">
                {historyError}
              </p>
            ) : null}

            {hasMoreHistory ? (
              <button
                type="button"
                onClick={() =>
                  void loadHistoryPage({
                    offset: historyOffset,
                    append: true,
                  })
                }
                disabled={isHistoryLoadingMore || isHistoryLoading}
                className="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold text-gray-600 transition hover:bg-gray-100 hover:text-gray-950 focus:outline-none focus:ring-2 focus:ring-gray-300 disabled:cursor-wait disabled:text-gray-400"
              >
                {isHistoryLoadingMore ? (
                  <Loader2 className="animate-spin" size={14} />
                ) : (
                  <History size={14} />
                )}
                Táº£i thÃªm
              </button>
            ) : null}
          </div>

          <div className="mt-6 space-y-2">
            <p className="px-2 text-xs font-medium uppercase tracking-wide text-gray-500">
              Demo cases
            </p>
            {demoCases.map((demo) => (
              <button
                key={demo.id}
                type="button"
                onClick={() => handleDemoClick(demo)}
                className="flex w-full items-start gap-2 rounded-lg px-3 py-2.5 text-left text-sm text-gray-700 transition hover:bg-gray-100 hover:text-gray-950 focus:outline-none focus:ring-2 focus:ring-gray-300"
              >
                <FileText className="mt-0.5 shrink-0" size={16} />
                {demo.label}
              </button>
            ))}
          </div>

          <div className="mt-auto space-y-2">
            <button
              type="button"
              onClick={onLogout}
              className="flex w-full items-center gap-2 rounded-lg px-3 py-2.5 text-left text-sm font-medium text-gray-700 transition hover:bg-gray-100 hover:text-gray-950 focus:outline-none focus:ring-2 focus:ring-gray-300"
            >
              <LogOut size={16} />
              Đăng xuất
            </button>
          </div>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="shrink-0 border-b border-gray-200 bg-white/95 px-4 py-4 backdrop-blur sm:px-6">
          <div className="mx-auto flex max-w-3xl flex-wrap items-center justify-between gap-3">
            <div className="min-w-0">
              <p className="text-base font-semibold text-gray-900">
                Clinical Auditor AI Agent
              </p>
            </div>
            <div className="flex min-w-0 items-center gap-2">
              <div className="flex rounded-xl border border-gray-200 bg-gray-50 p-1">
                <button
                  type="button"
                  onClick={() => setCenterMode('audit')}
                  className={`rounded-lg px-3 py-2 text-xs font-semibold transition focus:outline-none focus:ring-2 focus:ring-gray-300 ${
                    centerMode === 'audit'
                      ? 'bg-white text-gray-950 shadow-sm'
                      : 'text-gray-600 hover:text-gray-950'
                  }`}
                >
                  Kiểm tra đơn thuốc
                </button>
                <button
                  type="button"
                  onClick={() => setCenterMode('drug_chat')}
                  className={`rounded-lg px-3 py-2 text-xs font-semibold transition focus:outline-none focus:ring-2 focus:ring-gray-300 ${
                    centerMode === 'drug_chat'
                      ? 'bg-white text-gray-950 shadow-sm'
                      : 'text-gray-600 hover:text-gray-950'
                  }`}
                >
                  Hỏi về thuốc
                </button>
              </div>
              <button
                type="button"
                onClick={() => {
                  setIsMemoryPanelOpen(true)
                  setIsMemoryDrawerOpen(true)
                }}
                className={`items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-xs font-semibold text-gray-700 shadow-sm transition hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-300 ${
                  isMemoryPanelOpen ? 'inline-flex xl:hidden' : 'inline-flex'
                }`}
              >
                <PanelRightOpen size={15} />
                Doctor Memory
              </button>
              <button
                type="button"
                onClick={onLogout}
                className="inline-flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-xs font-semibold text-gray-700 shadow-sm transition hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-300 md:hidden"
              >
                <LogOut size={15} />
                Đăng xuất
              </button>
            </div>
          </div>
        </header>

        <div
          className={
            centerMode === 'audit' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'
          }
        >
        <section className="min-h-0 flex-1 overflow-y-auto bg-white">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center px-4">
              <div className="max-w-xl text-center">
                <h2 className="text-2xl font-semibold text-gray-900">
                  Kiểm tra đơn thuốc lâm sàng
                </h2>
                <p className="mt-3 text-sm leading-6 text-gray-500">
                  Nhập nội dung đơn thuốc hoặc chọn một ca demo để bắt đầu.
                </p>
                <div className="mt-6 grid gap-2 sm:grid-cols-2 md:hidden">
                  {demoCases.map((demo) => (
                    <button
                      key={demo.id}
                      type="button"
                      onClick={() => handleDemoClick(demo)}
                      className="rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm text-gray-700 transition hover:bg-gray-100 hover:text-gray-950"
                    >
                      {demo.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="py-3">
              {messages.map((message) => (
                <div
                  key={message.id}
                  ref={(element) => {
                    messageRefs.current[message.id] = element
                  }}
                >
                  <ChatMessage message={message} />
                </div>
              ))}
              <div ref={messageEndRef} />
            </div>
          )}
        </section>

        <ChatInput
          value={inputValue}
          onChange={handleInputChange}
          onSend={handleSend}
          disabled={isSubmitting}
          focusSignal={inputFocusSignal}
        />
        </div>

        <div
          className={
            centerMode === 'drug_chat'
              ? 'flex min-h-0 flex-1 flex-col'
              : 'hidden'
          }
        >
          <DrugChatPanel />
        </div>
      </main>

      {isMemoryPanelOpen ? (
        <div className="hidden w-[370px] shrink-0 border-l border-gray-200 xl:flex">
          <DoctorMemoryPanel
            latestAuditResult={latestAuditResult}
            relatedNotes={relatedMemoryNotes}
            onAddLocalNote={handleAddLocalMemoryNote}
            onClose={() => setIsMemoryPanelOpen(false)}
          />
        </div>
      ) : null}

      {isMemoryDrawerOpen ? (
        <div className="fixed inset-0 z-40 bg-black/20 xl:hidden">
          <div className="absolute inset-y-0 right-0 w-[min(390px,92vw)] border-l border-gray-200 bg-white shadow-2xl">
            <DoctorMemoryPanel
              latestAuditResult={latestAuditResult}
              relatedNotes={relatedMemoryNotes}
              onAddLocalNote={handleAddLocalMemoryNote}
              onClose={() => setIsMemoryDrawerOpen(false)}
            />
          </div>
        </div>
      ) : null}
    </div>
  )
}
