/**
 * Minimal Chart Legend Component
 * Shows only essential legend items for the glucose chart
 */

// Helper component for legend line items
function LegendLine({ color, dash = 'solid', width = 2 }: { color: string; dash?: 'solid' | 'dashed' | 'dotted'; width?: number }) {
  const dashArray = dash === 'dashed' ? '4,3' : dash === 'dotted' ? '2,2' : 'none'
  return (
    <svg width="16" height="8" className="flex-shrink-0">
      <line
        x1="0" y1="4" x2="16" y2="4"
        stroke={color}
        strokeWidth={width}
        strokeDasharray={dashArray}
      />
    </svg>
  )
}

interface ChartLegendProps {
  showEffectiveBg?: boolean
  showEffectAreas?: boolean
  showTftPredictions?: boolean
  showHistoricalIobCob?: boolean
}

export function ChartLegend({
  showEffectiveBg = false,
  showHistoricalIobCob = false,
}: ChartLegendProps = {}) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[11px]">
      {/* Primary glucose line */}
      <div className="flex items-center gap-1">
        <LegendLine color="#00c6ff" />
        <span className="text-gray-500">BG</span>
      </div>

      {/* Predicted BG (orange) */}
      <div className="flex items-center gap-1">
        <LegendLine color="#f97316" />
        <span className="text-gray-500">Predicted</span>
      </div>

      {/* BG Pressure gradient - only if enabled */}
      {showEffectiveBg && (
        <div className="flex items-center gap-1">
          <svg width="16" height="10" className="flex-shrink-0">
            <rect x="0" y="0" width="16" height="5" fill="#f97316" opacity="0.3" />
            <rect x="0" y="5" width="16" height="5" fill="#3b82f6" opacity="0.3" />
          </svg>
          <span className="text-gray-500">Pressure</span>
        </div>
      )}

      {/* IOB/COB/POB lines - only if enabled */}
      {showHistoricalIobCob && (
        <>
          <div className="flex items-center gap-1">
            <LegendLine color="#3b82f6" />
            <span className="text-gray-500">IOB</span>
          </div>
          <div className="flex items-center gap-1">
            <LegendLine color="#22c55e" />
            <span className="text-gray-500">COB</span>
          </div>
          <div className="flex items-center gap-1">
            <LegendLine color="#a855f7" />
            <span className="text-gray-500">POB</span>
          </div>
        </>
      )}
    </div>
  )
}

export default ChartLegend
