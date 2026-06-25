import { useEffect, useRef, useState } from 'react'
import { FileText, Plus } from 'lucide-react'
import ChatInput from './ChatInput'
import ChatMessage from './ChatMessage'
import type { ChatMessageItem } from '../types/chat'

const metforminDemo = `Metformin (Panfor SR) 750mg x 30 viên
Omeprazol (Kagascdine) 20mg x 14 viên
Paracetamol (Hapacol) 500mg x 10 viên`

const clopidogrelDemo = `Clopidogrel (Plavix) 75mg x 30 viên
Omeprazol (Kagascdine) 20mg x 14 viên
Atorvastatin 20mg x 30 viên`

const fakeAssistantResponse =
  'Đã nhận đơn thuốc. Bước tiếp theo sẽ kết nối API kiểm tra đơn thuốc.'

const createMessageId = () => crypto.randomUUID()

export default function ChatLayout() {
  const [messages, setMessages] = useState<ChatMessageItem[]>([])
  const [inputValue, setInputValue] = useState('')
  const messageEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const handleSend = () => {
    const trimmedInput = inputValue.trim()

    if (!trimmedInput) {
      return
    }

    const userMessage: ChatMessageItem = {
      id: createMessageId(),
      role: 'user',
      content: trimmedInput,
    }

    const assistantMessage: ChatMessageItem = {
      id: createMessageId(),
      role: 'assistant',
      content: fakeAssistantResponse,
      riskLevel: 'high',
    }

    setMessages((currentMessages) => [
      ...currentMessages,
      userMessage,
      assistantMessage,
    ])
    setInputValue('')
  }

  const handleNewAudit = () => {
    setMessages([])
    setInputValue('')
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
            <button
              type="button"
              onClick={() => setInputValue(metforminDemo)}
              className="flex w-full items-start gap-2 rounded-lg px-3 py-2.5 text-left text-sm text-gray-700 transition hover:bg-gray-100 hover:text-gray-950 focus:outline-none focus:ring-2 focus:ring-gray-300"
            >
              <FileText className="mt-0.5 shrink-0" size={16} />
              Metformin + eGFR 25
            </button>
            <button
              type="button"
              onClick={() => setInputValue(clopidogrelDemo)}
              className="flex w-full items-start gap-2 rounded-lg px-3 py-2.5 text-left text-sm text-gray-700 transition hover:bg-gray-100 hover:text-gray-950 focus:outline-none focus:ring-2 focus:ring-gray-300"
            >
              <FileText className="mt-0.5 shrink-0" size={16} />
              Clopidogrel + Omeprazole
            </button>
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
                  <button
                    type="button"
                    onClick={() => setInputValue(metforminDemo)}
                    className="rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm text-gray-700 transition hover:bg-gray-100 hover:text-gray-950"
                  >
                    Metformin + eGFR 25
                  </button>
                  <button
                    type="button"
                    onClick={() => setInputValue(clopidogrelDemo)}
                    className="rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm text-gray-700 transition hover:bg-gray-100 hover:text-gray-950"
                  >
                    Clopidogrel + Omeprazole
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-3">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              <div ref={messageEndRef} />
            </div>
          )}
        </section>

        <ChatInput
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSend}
        />
      </main>
    </div>
  )
}
