import type { ChatSource } from '../types/chat'

type ChatSourceReferencesProps = {
  sources?: ChatSource[] | null
}

const hasText = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0

const sourceLabel = (source: ChatSource): string =>
  [
    source.title,
    source.label,
    source.name,
    source.source,
  ].find(hasText)?.trim() || 'Nguồn tham khảo'

const sourceSection = (source: ChatSource): string =>
  [source.section].find(hasText)?.trim() || ''

const sourceSnippet = (source: ChatSource): string =>
  [source.snippet, source.text].find(hasText)?.trim() || ''

const sourceKey = (source: ChatSource, index: number): string => {
  const label = sourceLabel(source)
  const section = sourceSection(source)
  return `${source.url || label}-${section}-${index}`
}

export default function ChatSourceReferences({
  sources,
}: ChatSourceReferencesProps) {
  const visibleSources = Array.isArray(sources) ? sources : []

  if (visibleSources.length === 0) {
    return null
  }

  return (
    <section className="mt-4 rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 text-xs leading-5 text-gray-600">
      <h4 className="font-semibold text-gray-800">Nguồn tham khảo</h4>
      <ol className="mt-2 space-y-2">
        {visibleSources.map((source, index) => {
          const label = sourceLabel(source)
          const section = sourceSection(source)
          const snippet = sourceSnippet(source)

          return (
            <li key={sourceKey(source, index)}>
              <div>
                <span className="font-medium text-gray-500">[{index + 1}]</span>{' '}
                <span>{label}</span>
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
              </div>
              {section ? <p className="mt-0.5 text-gray-500">{section}</p> : null}
              {snippet ? (
                <p className="mt-1 line-clamp-3 text-gray-600">{snippet}</p>
              ) : null}
            </li>
          )
        })}
      </ol>
    </section>
  )
}
