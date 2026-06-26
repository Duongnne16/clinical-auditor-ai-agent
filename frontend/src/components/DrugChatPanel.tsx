import { Send, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { sendChatMessage } from '../api/chatApi'
import type { ChatIntent, DrugChatMessageItem } from '../types/chat'
import ChatSourceReferences from './ChatSourceReferences'

const createMessageId = () => crypto.randomUUID()

const exampleQuestions = [
  'Omeprazole có tương tác với Clopidogrel không?',
  'Paracetamol có tác dụng phụ gì?',
  'Levofloxacin dùng cần lưu ý gì?',
  'Viết giúp tôi bài văn',
]

const intentLabels: Record<string, string> = {
  drug_interaction_query: 'Tương tác thuốc',
  single_drug_query: 'Thông tin thuốc',
  out_of_scope: 'Ngoài phạm vi',
}

const warningLabel = (warning: string): string => {
  const labels: Record<string, string> = {
    insufficient_evidence: 'Chưa tìm thấy đủ bằng chứng tham khảo phù hợp.',
    missing_interaction_drug_mentions:
      'Cần nhập rõ các thuốc cần rà soát tương tác.',
    single_drug_query_requires_drug_name: 'Cần nhập rõ tên thuốc cần tra cứu.',
  }

  return labels[warning] || warning
}

function IntentBadge({ intent }: { intent?: ChatIntent }) {
  if (!intent) {
    return null
  }

  return (
    <span className="inline-flex rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">
      {intentLabels[intent] || 'Tra cứu thuốc'}
    </span>
  )
}

function AssistantMessage({ message }: { message: DrugChatMessageItem }) {
  const response = message.response
  const warnings = response?.warnings || []

  return (
    <div className="px-4 py-5 sm:px-6">
      <div className="mx-auto flex w-full max-w-3xl gap-4">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-emerald-200 bg-emerald-50 text-sm font-semibold text-emerald-700">
          AI
        </div>
        <div className="min-w-0 flex-1 text-sm leading-7 text-gray-900">
          {message.status === 'loading' ? (
            <span className="inline-flex items-center gap-2">
              <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
              {message.content}
            </span>
          ) : (
            <div className={message.status === 'error' ? 'text-red-700' : ''}>
              {response?.intent ? (
                <div className="mb-3">
                  <IntentBadge intent={response.intent} />
                </div>
              ) : null}

              <div className="whitespace-pre-wrap">{message.content}</div>

              {warnings.length > 0 ? (
                <div className="mt-4 space-y-2">
                  {warnings.map((warning) => (
                    <div
                      key={warning}
                      className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-900"
                    >
                      {warningLabel(warning)}
                    </div>
                  ))}
                </div>
              ) : null}

              {response?.disclaimer ? (
                <p className="mt-4 rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 text-xs leading-5 text-gray-600">
                  {response.disclaimer}
                </p>
              ) : null}

              <ChatSourceReferences sources={response?.sources} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end px-4 py-3 sm:px-6">
      <div className="max-w-[min(42rem,85%)] whitespace-pre-wrap rounded-3xl bg-gray-100 px-5 py-3 text-sm leading-6 text-gray-900 shadow-sm">
        {content}
      </div>
    </div>
  )
}

export default function DrugChatPanel() {
  const [messages, setMessages] = useState<DrugChatMessageItem[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const messageEndRef = useRef<HTMLDivElement | null>(null)
  const canSend = inputValue.trim().length > 0 && !isSubmitting

  useEffect(() => {
    const textarea = textareaRef.current

    if (!textarea) {
      return
    }

    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`
    textarea.style.overflowY = textarea.scrollHeight > 180 ? 'auto' : 'hidden'
  }, [inputValue])

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const replaceAssistantMessage = (
    messageId: string,
    nextMessage: Partial<DrugChatMessageItem>,
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
    const userMessage: DrugChatMessageItem = {
      id: createMessageId(),
      role: 'user',
      content: trimmedInput,
    }
    const loadingMessage: DrugChatMessageItem = {
      id: loadingMessageId,
      role: 'assistant',
      content: 'Đang tra cứu...',
      status: 'loading',
    }

    setMessages((currentMessages) => [
      ...currentMessages,
      userMessage,
      loadingMessage,
    ])
    setInputValue('')
    setIsSubmitting(true)

    try {
      const response = await sendChatMessage(trimmedInput)
      replaceAssistantMessage(loadingMessageId, {
        content: response.answer || response.message || '',
        status: undefined,
        response,
      })
    } catch (error) {
      replaceAssistantMessage(loadingMessageId, {
        content:
          error instanceof Error
            ? error.message
            : 'Không thể tra cứu thuốc do lỗi không xác định.',
        status: 'error',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey) {
      return
    }

    event.preventDefault()

    if (canSend) {
      handleSend()
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-white">
      <section className="min-h-0 flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="mx-auto flex min-h-full max-w-3xl flex-col justify-center px-4 py-8 sm:px-6">
            <div>
              <h2 className="text-2xl font-semibold text-gray-900">
                Hỏi nhanh về thuốc
              </h2>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-gray-500">
                Tra cứu tương tác, lưu ý sử dụng và thông tin thuốc từ dữ liệu
                tham khảo.
              </p>

              <div className="mt-6 flex flex-wrap gap-2">
                {exampleQuestions.map((question) => (
                  <button
                    key={question}
                    type="button"
                    onClick={() => {
                      setInputValue(question)
                      textareaRef.current?.focus()
                    }}
                    className="rounded-full border border-gray-200 bg-white px-3 py-2 text-left text-xs font-medium text-gray-700 transition hover:bg-gray-100 hover:text-gray-950 focus:outline-none focus:ring-2 focus:ring-gray-300"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="py-3">
            {messages.map((message) =>
              message.role === 'user' ? (
                <UserMessage key={message.id} content={message.content} />
              ) : (
                <AssistantMessage key={message.id} message={message} />
              ),
            )}
            <div ref={messageEndRef} />
          </div>
        )}
      </section>

      <div className="border-t border-gray-200 bg-white px-4 py-4 sm:px-6">
        <div className="mx-auto max-w-3xl">
          {messages.length > 0 ? (
            <div className="mb-3 flex justify-end">
              <button
                type="button"
                onClick={() => setMessages([])}
                disabled={isSubmitting}
                className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold text-gray-600 transition hover:bg-gray-100 hover:text-gray-950 focus:outline-none focus:ring-2 focus:ring-gray-300 disabled:cursor-not-allowed disabled:text-gray-400"
              >
                <Trash2 size={15} />
                Xóa hội thoại
              </button>
            </div>
          ) : null}

          <div className="flex items-end gap-2 rounded-3xl border border-gray-300 bg-white p-2 shadow-sm">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(event) => setInputValue(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Nhập câu hỏi về thuốc, ví dụ: Omeprazole có tương tác với Clopidogrel không?"
              rows={1}
              className="min-h-12 flex-1 resize-none bg-transparent px-3 py-3 text-sm leading-6 text-gray-900 placeholder:text-gray-500 focus:outline-none"
            />

            <button
              type="button"
              onClick={handleSend}
              disabled={!canSend}
              className="inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-full border border-gray-300 bg-gray-100 px-4 text-sm font-semibold text-gray-900 transition hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-300 disabled:cursor-not-allowed disabled:border-gray-200 disabled:bg-gray-100 disabled:text-gray-400"
            >
              <Send size={17} />
              Gửi
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
