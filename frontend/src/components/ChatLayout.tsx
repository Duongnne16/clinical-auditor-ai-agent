import { useEffect, useRef, useState } from 'react'
import { FileText, PanelRightOpen, Plus } from 'lucide-react'
import { auditPrescription } from '../api/prescriptionAuditApi'
import ChatInput from './ChatInput'
import ChatMessage from './ChatMessage'
import DoctorMemoryPanel from './DoctorMemoryPanel'
import type { ChatMessageItem } from '../types/chat'
import type { DoctorMemoryNote } from '../types/doctorNote'
import type {
  PatientContext,
  PrescriptionAuditRequest,
  PrescriptionAuditResponse,
} from '../types/prescriptionAudit'

type DemoCase = {
  id: string
  label: string
  prescriptionText: string
  patientContext: PatientContext
  queryTypes: string[]
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

export default function ChatLayout() {
  const [messages, setMessages] = useState<ChatMessageItem[]>([])
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
  const messageEndRef = useRef<HTMLDivElement | null>(null)
  const messageRefs = useRef<Record<string, HTMLDivElement | null>>({})

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
      doctor_id: 'dev-doctor-001',
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
    nextMessage: Partial<ChatMessageItem>,
  ) => {
    setMessages((currentMessages) =>
      currentMessages.map((message) =>
        message.id === messageId ? { ...message, ...nextMessage } : message,
      ),
    )
  }

  const handleSend = async () => {
    const trimmedInput = inputValue.trim()

    if (!trimmedInput || isSubmitting) {
      return
    }

    const loadingMessageId = createMessageId()
    const userMessage: ChatMessageItem = {
      id: createMessageId(),
      role: 'user',
      content: trimmedInput,
    }
    const loadingMessage: ChatMessageItem = {
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

          <p className="mt-auto rounded-lg bg-gray-100 px-3 py-3 text-xs leading-5 text-gray-500">
            OCR/PDF upload coming later
          </p>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="shrink-0 border-b border-gray-200 bg-white/95 px-4 py-4 backdrop-blur sm:px-6">
          <div className="mx-auto flex max-w-3xl items-center justify-between">
            <div>
              <p className="text-base font-semibold text-gray-900">
                Clinical Auditor AI Agent
              </p>
              <p className="text-xs text-gray-500 md:hidden">
                OCR/PDF upload coming later
              </p>
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
          </div>
        </header>

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
