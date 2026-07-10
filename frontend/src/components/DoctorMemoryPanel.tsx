import { Brain, PanelRightClose, Save } from 'lucide-react'
import { useEffect, useState } from 'react'
import { createDoctorNote } from '../api/doctorNotes'
import type { DoctorMemoryNote } from '../types/doctorNote'
import type { PrescriptionAuditResponse } from '../types/prescriptionAudit'
import { buildDoctorNotePayload } from '../utils/doctorMemoryContext'

type DoctorMemoryPanelProps = {
  latestAuditResult: PrescriptionAuditResponse | null
  relatedNotes: DoctorMemoryNote[]
  onAddLocalNote: (note: DoctorMemoryNote) => void
  onClose: () => void
}

const noteTypeLabels: Record<string, string> = {
  clinical_experience: 'Kinh nghiệm lâm sàng',
  drug_interaction_note: 'Tương tác thuốc',
  contraindication_note: 'Chống chỉ định',
  monitoring_note: 'Theo dõi',
  dose_context_note: 'Bối cảnh liều',
}

const noteKey = (note: DoctorMemoryNote, index: number): string =>
  String(note.note_id || note.id || `${note.title || 'note'}-${index}`)

const noteTitle = (note: DoctorMemoryNote): string =>
  note.title?.trim() || 'Ghi chú riêng'

const noteText = (note: DoctorMemoryNote): string =>
  note.note_text?.trim() || note.content?.trim() || ''

const MIN_LIGHTWEIGHT_NOTE_LENGTH = 16
const NOTE_VALIDATION_MESSAGE =
  'Ghi chÃº quÃ¡ ngáº¯n hoáº·c chÆ°a Ä‘á»§ ná»™i dung chuyÃªn mÃ´n Ä‘á»ƒ lÆ°u.'

const formatDate = (value?: string | null): string => {
  if (!value) {
    return ''
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return ''
  }

  return new Intl.DateTimeFormat('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  }).format(date)
}

function EmptyState({ hasAuditContext }: { hasAuditContext: boolean }) {
  if (!hasAuditContext) {
    return (
      <div className="rounded-xl border border-dashed border-gray-200 bg-gray-50 px-4 py-5 text-sm leading-6 text-gray-600">
        <p className="font-medium text-gray-800">Chưa có ngữ cảnh đơn thuốc.</p>
        <p className="mt-1">
          Sau khi kiểm tra đơn, các ghi chú liên quan sẽ xuất hiện tại đây.
        </p>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-4 text-sm leading-6 text-gray-600">
      Chưa có ghi chú liên quan đến đơn thuốc này.
    </div>
  )
}

function NoteCard({ note, index }: { note: DoctorMemoryNote; index: number }) {
  const typeLabel = note.note_type
    ? noteTypeLabels[note.note_type] || note.note_type
    : ''
  const createdAt = formatDate(note.created_at)
  const text = noteText(note)

  return (
    <article
      className="rounded-xl border border-blue-100 bg-white px-4 py-3 shadow-sm"
      key={noteKey(note, index)}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="break-words text-sm font-semibold text-gray-950">
            {noteTitle(note)}
          </p>
          <div className="mt-1 flex flex-wrap gap-1.5 text-[11px] font-medium">
            <span className="rounded-full bg-blue-50 px-2 py-0.5 text-blue-700">
              Ghi chú riêng
            </span>
            {note.is_newly_saved ? (
              <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-emerald-700">
                Vừa lưu
              </span>
            ) : null}
          </div>
        </div>
      </div>

      {text ? (
        <p className="mt-3 whitespace-pre-wrap break-words text-sm leading-6 text-gray-700">
          {text}
        </p>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-500">
        {typeLabel ? <span>{typeLabel}</span> : null}
        {createdAt ? <span>{createdAt}</span> : null}
        {note.match_reason && !note.is_newly_saved ? (
          <span>{note.match_reason}</span>
        ) : null}
      </div>
    </article>
  )
}

export default function DoctorMemoryPanel({
  latestAuditResult,
  relatedNotes,
  onAddLocalNote,
  onClose,
}: DoctorMemoryPanelProps) {
  const [noteTextValue, setNoteTextValue] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const hasAuditContext = Boolean(latestAuditResult)
  const trimmedNote = noteTextValue.trim()
  const canSave = hasAuditContext && trimmedNote.length > 0 && !isSaving

  useEffect(() => {
    if (!statusMessage) {
      return
    }

    const timeoutId = window.setTimeout(() => setStatusMessage(''), 3500)
    return () => window.clearTimeout(timeoutId)
  }, [statusMessage])

  const handleSave = async () => {
    if (!latestAuditResult || !canSave) {
      return
    }
    if (trimmedNote.length < MIN_LIGHTWEIGHT_NOTE_LENGTH) {
      setStatusMessage('')
      setErrorMessage(NOTE_VALIDATION_MESSAGE)
      return
    }

    const payload = buildDoctorNotePayload(latestAuditResult, trimmedNote)
    setIsSaving(true)
    setStatusMessage('')
    setErrorMessage('')

    try {
      const savedNote = await createDoctorNote(payload)
      onAddLocalNote({
        ...payload,
        ...savedNote,
        title: payload.title,
        note_text: payload.note_text,
        content: payload.content,
        active_ingredients: payload.active_ingredients,
        drug_pair_keys: payload.drug_pair_keys,
        diagnosis_keywords: payload.diagnosis_keywords,
        patient_tags: payload.patient_tags,
        match_reason: 'newly_saved',
        is_newly_saved: true,
      })
      setNoteTextValue('')
      setStatusMessage('Đã lưu ghi chú.')
    } catch (error) {
      setErrorMessage('Không lưu được ghi chú. Vui lòng thử lại.')
      if (error instanceof Error && error.message) {
        setErrorMessage(error.message)
      }
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <aside className="flex h-full min-h-0 w-full flex-col bg-white">
      <header className="shrink-0 border-b border-gray-200 px-4 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-700">
              <Brain size={18} />
            </span>
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-gray-950">
                Doctor Memory
              </h2>
              <p className="mt-1 text-xs leading-5 text-gray-500">
                Ghi chú chuyên môn riêng của bạn
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-gray-500 transition hover:bg-gray-100 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-300"
            aria-label="Collapse Doctor Memory"
            title="Collapse Doctor Memory"
          >
            <PanelRightClose size={17} />
          </button>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <section>
          <h3 className="text-sm font-semibold text-gray-950">
            Ghi chú liên quan
          </h3>
          <div className="mt-3 space-y-3">
            {relatedNotes.length > 0 ? (
              relatedNotes.map((note, index) => (
                <NoteCard key={noteKey(note, index)} note={note} index={index} />
              ))
            ) : (
              <EmptyState hasAuditContext={hasAuditContext} />
            )}
          </div>
        </section>

        <section className="mt-6 border-t border-gray-200 pt-5">
          <h3 className="text-sm font-semibold text-gray-950">
            Thêm ghi chú mới
          </h3>
          <p className="mt-1 text-xs leading-5 text-gray-500">
            {hasAuditContext
              ? 'Ghi chú sẽ được lưu cùng ngữ cảnh đơn thuốc hiện tại.'
              : 'Hãy kiểm tra đơn thuốc trước khi thêm ghi chú theo ngữ cảnh.'}
          </p>

          <textarea
            value={noteTextValue}
            onChange={(event) => setNoteTextValue(event.target.value)}
            disabled={!hasAuditContext || isSaving}
            placeholder="Nhập ghi chú chuyên môn của bạn..."
            rows={5}
            className="mt-3 w-full resize-none rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-sm leading-6 text-gray-900 shadow-sm outline-none transition placeholder:text-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-gray-50 disabled:text-gray-500"
          />

          {statusMessage ? (
            <p className="mt-2 text-xs font-medium text-emerald-700">
              {statusMessage}
            </p>
          ) : null}
          {errorMessage ? (
            <p className="mt-2 text-xs font-medium text-red-700">
              {errorMessage}
            </p>
          ) : null}

          <button
            type="button"
            onClick={handleSave}
            disabled={!canSave}
            className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-200 disabled:cursor-not-allowed disabled:bg-gray-300"
          >
            <Save size={16} />
            {isSaving ? 'Đang lưu...' : 'Lưu ghi chú'}
          </button>
        </section>
      </div>
    </aside>
  )
}
