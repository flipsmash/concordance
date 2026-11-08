import { useEffect, useRef, useState } from 'react'
import './GrowthChart.css'

const PLOT_HEIGHT = 90 // px
const PADDING_X = 16 // px -- room for the first/last point's own hit-circle and label to sit inside the plot
const LABEL_MIN_GAP = 64 // px between date labels: as many as fit the width, never crowded

// A real time series (day on the x-axis), unlike DifficultyHistogram's
// value-range buckets -- and deliberately NOT built on that component: its
// bars are buttons with a click-to-filter affordance, which doesn't apply
// here (a single day isn't a filterable range). A line, not bars: both the
// daily and cumulative views read as a trend to follow across the whole
// range, and a cumulative bar chart in particular (every bar taller than
// the last, by construction) is a shape that's easy to mistake for meaning
// something -- a plain ascending line is the more honest rendering of "this
// is just a running total." One hand-rolled SVG path rather than a chart
// library: this is one geometry (a polyline + area fill) reused three
// times, not worth a dependency for.
function formatShort(iso) {
  const d = new Date(`${iso}T00:00:00`)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function GrowthChart({ data, unit }) {
  // Sized to its container (not a fixed 10px per day, which overflowed on a
  // phone): x positions are computed from the measured width.
  const wrapRef = useRef(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return undefined
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  // The measured container renders in every state: an early return here
  // left the ResizeObserver with nothing to observe, so width stayed 0.
  if (data === null || data.length === 0) {
    return (
      <div className="growth-chart growth-chart-loading muted" ref={wrapRef}>
        {data === null ? 'Loading…' : 'No data yet.'}
      </div>
    )
  }
  const maxValue = Math.max(...data.map((d) => d.value), 1)
  const lastIndex = data.length - 1
  const w = Math.max(width, PADDING_X * 2 + 1)
  const step = lastIndex ? (w - PADDING_X * 2) / lastIndex : 0
  const labelEvery = Math.max(1, Math.ceil(LABEL_MIN_GAP / Math.max(step, 0.01)))

  function x(i) {
    return PADDING_X + i * step
  }
  function y(v) {
    return PLOT_HEIGHT - (v / maxValue) * PLOT_HEIGHT
  }

  const points = data.map((d, i) => [x(i), y(d.value)])
  const linePoints = points.map(([px, py]) => `${px},${py}`).join(' ')
  const areaPath =
    `M${points[0][0]},${PLOT_HEIGHT} ` +
    points.map(([px, py]) => `L${px},${py}`).join(' ') +
    ` L${points[lastIndex][0]},${PLOT_HEIGHT} Z`

  return (
    <div className="growth-chart" ref={wrapRef}>
      {width > 0 && (
        <>
          <svg className="growth-chart-svg" width={w} height={PLOT_HEIGHT} viewBox={`0 0 ${w} ${PLOT_HEIGHT}`}>
            <line x1={0} y1={PLOT_HEIGHT} x2={w} y2={PLOT_HEIGHT} className="growth-chart-baseline" />
            <path d={areaPath} className="growth-chart-area" />
            <polyline points={linePoints} className="growth-chart-line" />
            {data.map((d, i) => (
              <circle key={d.date} cx={x(i)} cy={y(d.value)} r={Math.max(3, Math.min(6, step / 2))} className="growth-chart-hit">
                <title>{`${formatShort(d.date)}: ${d.value.toLocaleString()} ${unit}`}</title>
              </circle>
            ))}
          </svg>
          <div className="growth-chart-labels">
            {data.map((d, i) => {
              // The last point always gets a label; a periodic one too close
              // to it would collide with that label's text, so it's skipped.
              const showLabel =
                i === 0 || i === lastIndex || (i % labelEvery === 0 && (lastIndex - i) * step >= LABEL_MIN_GAP)
              if (!showLabel) return null
              return (
                <span key={d.date} className="growth-chart-label" style={{ left: x(i) }}>
                  {formatShort(d.date)}
                </span>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}

export default GrowthChart
