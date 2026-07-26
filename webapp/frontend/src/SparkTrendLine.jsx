import { useMemo, useState } from 'react'
import { cssVar } from './graphUtils'

const VIEW_W = 600
const VIEW_H = 160
const PAD_X = 12
const PAD_Y = 16

/** Score-over-time trend line for the progress dashboard. x is the session's
 * ordinal position (NOT real elapsed time) -- sessions aren't evenly spaced
 * in time, and a true time axis would leave huge dead gaps between bursts of
 * practice. y is a fixed 0-100 axis (score_pct). Precomputed/static data, so
 * plain SVG with onMouseEnter hit-testing (same rationale as
 * AuthorClusterMap.jsx) rather than canvas or a simulation. */
function SparkTrendLine({ points }) {
  const [hoverIdx, setHoverIdx] = useState(null)

  const { coords } = useMemo(() => {
    if (!points || points.length === 0) return { coords: [] }
    const n = points.length
    const stepX = n > 1 ? (VIEW_W - 2 * PAD_X) / (n - 1) : 0
    return {
      coords: points.map((p, i) => ({
        x: PAD_X + i * stepX,
        y: VIEW_H - PAD_Y - (p.score_pct / 100) * (VIEW_H - 2 * PAD_Y),
        point: p,
      })),
    }
  }, [points])

  if (!points || points.length === 0) {
    return <p className="progress-empty-state">Take a quiz to start your trend line.</p>
  }

  const accent = cssVar('--accent', '#1c6dbd')
  const path = coords.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c.x.toFixed(1)} ${c.y.toFixed(1)}`).join(' ')
  const hovered = hoverIdx !== null ? coords[hoverIdx] : null

  return (
    <div className="spark-trend-wrap">
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="spark-trend-svg" role="img" aria-label="Score over time">
        {/* baseline gridlines at 0/50/100% for scale context */}
        {[0, 50, 100].map((pct) => {
          const y = VIEW_H - PAD_Y - (pct / 100) * (VIEW_H - 2 * PAD_Y)
          return (
            <line key={pct} x1={PAD_X} x2={VIEW_W - PAD_X} y1={y} y2={y}
                  stroke={cssVar('--border', '#e5e4e7')} strokeWidth="1" />
          )
        })}
        {coords.length === 1 ? (
          <circle cx={coords[0].x} cy={coords[0].y} r="4" fill={accent} />
        ) : (
          <path d={path} fill="none" stroke={accent} strokeWidth="2" />
        )}
        {coords.map((c, i) => (
          <circle
            key={c.point.session_id}
            cx={c.x}
            cy={c.y}
            r={hoverIdx === i ? 6 : 4}
            fill={accent}
            opacity={hoverIdx === null || hoverIdx === i ? 1 : 0.4}
            onMouseEnter={() => setHoverIdx(i)}
            onMouseLeave={() => setHoverIdx((h) => (h === i ? null : h))}
          />
        ))}
        {hovered && (
          <line x1={hovered.x} x2={hovered.x} y1={PAD_Y / 2} y2={VIEW_H - PAD_Y}
                stroke={cssVar('--text', '#6b6375')} strokeWidth="1" strokeDasharray="3,3" opacity="0.5" />
        )}
      </svg>
      {hovered && (
        <div className="spark-trend-tooltip">
          {new Date(hovered.point.finished_at).toLocaleDateString()} — {hovered.point.score_pct.toFixed(0)}%
          {' '}({hovered.point.total_questions} question{hovered.point.total_questions === 1 ? '' : 's'})
        </div>
      )}
    </div>
  )
}

export default SparkTrendLine
