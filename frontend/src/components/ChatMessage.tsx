import type { ChatMessageItem } from '../types/chat'
import RiskBadge from './RiskBadge'

type ChatMessageProps = {
  message: ChatMessageItem
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
          {message.riskLevel ? (
            <div className="mb-3">
              <RiskBadge level={message.riskLevel} />
            </div>
          ) : null}
          <div className="whitespace-pre-wrap">{message.content}</div>
        </div>
      </div>
    </div>
  )
}
