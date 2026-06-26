import type { RiskLevel } from '../types/chat'

type RiskBadgeProps = {
  level?: RiskLevel | string | null
}

const badgeStyles: Record<RiskLevel, string> = {
  high: 'border-red-200 bg-red-50 text-red-700',
  moderate: 'border-orange-200 bg-orange-50 text-orange-700',
  low: 'border-green-200 bg-green-50 text-green-700',
  unknown: 'border-gray-200 bg-gray-50 text-gray-700',
}
const badgeLabels: Record<RiskLevel, string> = {
  high: 'Cao',
  moderate: 'Trung bình',
  low: 'Thấp',
  unknown: 'Cần bổ sung thông tin',
}

const normalizeRiskLevel = (level?: RiskLevel | string | null): RiskLevel => {
  if (level === 'high' || level === 'moderate' || level === 'low') {
    return level
  }

  return 'unknown'
}

export default function RiskBadge({ level }: RiskBadgeProps) {
  const normalizedLevel = normalizeRiskLevel(level)

  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold uppercase tracking-wide ${badgeStyles[normalizedLevel]}`}
    >
      {badgeLabels[normalizedLevel]}
    </span>
  )
}
