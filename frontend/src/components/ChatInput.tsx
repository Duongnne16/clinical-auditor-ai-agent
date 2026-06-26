import { useEffect, useRef, type KeyboardEvent } from 'react'
import { Plus, Send } from 'lucide-react'

type ChatInputProps = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  disabled?: boolean
  focusSignal?: number
}

export default function ChatInput({
  value,
  onChange,
  onSend,
  disabled = false,
  focusSignal = 0,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const canSend = value.trim().length > 0 && !disabled

  useEffect(() => {
    const textarea = textareaRef.current

    if (!textarea) {
      return
    }

    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`
    textarea.style.overflowY = textarea.scrollHeight > 180 ? 'auto' : 'hidden'
  }, [value])

  useEffect(() => {
    if (focusSignal > 0) {
      textareaRef.current?.focus()
    }
  }, [focusSignal])

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey) {
      return
    }

    event.preventDefault()

    if (canSend) {
      onSend()
    }
  }

  const handleUploadClick = () => {
    alert('Tính năng upload ảnh/PDF sẽ được phát triển ở phiên bản tiếp theo.')
  }

  return (
    <div className="border-t border-gray-200 bg-white px-4 py-4 sm:px-6">
      <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-3xl border border-gray-300 bg-white p-2 shadow-sm">
        <button
          type="button"
          onClick={handleUploadClick}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-gray-600 transition hover:bg-gray-100 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-300"
          aria-label="Upload image or PDF"
          title="Upload image or PDF"
        >
          <Plus size={20} />
        </button>

        <textarea
          ref={textareaRef}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Nhập đơn thuốc cần kiểm tra..."
          rows={1}
          className="min-h-12 flex-1 resize-none bg-transparent px-1 py-3 text-sm leading-6 text-gray-900 placeholder:text-gray-500 focus:outline-none"
        />

        <button
          type="button"
          onClick={onSend}
          disabled={!canSend}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-gray-300 bg-gray-100 text-gray-900 transition hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-300 disabled:cursor-not-allowed disabled:border-gray-200 disabled:bg-gray-100 disabled:text-gray-400"
          aria-label="Send message"
          title="Send message"
        >
          <Send size={18} />
        </button>
      </div>
    </div>
  )
}
