import { useEffect, useMemo, useRef, useState } from 'react'
import './NgramTrendChart.css'

// Popularity-over-time line chart for 1-4 terms, from /api/browse/ngram-trend
// (per-decade Google Books counts, occurrences per million words). One shared
// y-axis; slot colors are the validated categorical order (blue, orange,
// aqua, yellow -- see NgramTrendChart.css), always by input position so a
// term keeps its color as others are added/removed. Identity never rides
// on color alone: legend + direct end-of-line labels + a table view.

// Laid out at the container's real pixel width (not a scaled viewBox), so
// text stays legible on a phone instead of shrinking with the drawing.
const H = 300
const M = { top: 16, bottom: 34, left: 52 }
const PLOT_H = H - M.top - M.bottom
const LABEL_GAP = 15
const MIN_X_LABEL_PX = 44       // decade labels are thinned until they're at least this far apart

function niceMax(v) {
  if (v <= 0) return 1
  const p = 10 ** Math.floor(Math.log10(v))
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((c) => c >= v)
}

function formatRate(v) {
  if (v === 0) return '0'
  if (v >= 100) return Math.round(v).toLocaleString()
  if (v >= 1) return v.toFixed(1)
  return v.toPrecision(2)
}

function NgramTrendChart({ data, logScale = false }) {
  const [hover, setHover] = useState(null) // decade index
  const [W, setW] = useState(760)
  const svgRef = useRef(null)
  const plotRef = useRef(null)
  const { decades, series } = data

  useEffect(() => {
    const el = plotRef.current
    if (!el) return undefined
    const ro = new ResizeObserver(([entry]) => setW(Math.max(300, Math.round(entry.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const right = W < 520 ? 84 : 118     // room for the direct end-of-line labels
  const PLOT_W = W - M.left - right
  const shown = useMemo(() => series.filter((s) => s.found), [series])

  const scale = useMemo(() => {
    const values = shown.flatMap((s) => s.per_million)
    const positive = values.filter((v) => v > 0)
    if (logScale && positive.length) {
      const lo = 10 ** Math.floor(Math.log10(Math.min(...positive)))
      const hi = 10 ** Math.ceil(Math.log10(Math.max(...positive)))
      const ticks = []
      for (let t = lo; t <= hi * 1.0001; t *= 10) ticks.push(t)
      const y = (v) => (v > 0 ? M.top + PLOT_H * (1 - Math.log10(v / lo) / Math.log10(hi / lo || 10)) : null)
      return { y, ticks }
    }
    const max = niceMax(Math.max(0, ...values))
    return { y: (v) => M.top + PLOT_H * (1 - v / max), ticks: [0, max / 4, max / 2, (3 * max) / 4, max] }
  }, [shown, logScale])

  // Label every Nth decade (20 years on a wide chart), always ending clear of
  // the right edge rather than crowding a partial last label in.
  const step = decades.length > 1 ? PLOT_W / (decades.length - 1) : PLOT_W
  const every = [2, 4, 5, 10].find((n) => n * step >= MIN_X_LABEL_PX) || 10

  const x = (i) => M.left + (decades.length === 1 ? 0 : (i / (decades.length - 1)) * PLOT_W)

  const paths = shown.map((s) => {
    let d = ''
    let pen = false
    s.per_million.forEach((v, i) => {
      const yy = scale.y(v)
      if (yy === null) { pen = false; return }       // log scale: a zero decade is a gap, not a plunge
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${yy.toFixed(1)} `
      pen = true
    })
    return d
  })

  // Direct labels at the right end, nudged apart so they never overlap.
  const endLabels = useMemo(() => {
    const items = shown.map((s, k) => {
      let i = s.per_million.length - 1
      while (i > 0 && scale.y(s.per_million[i]) === null) i -= 1
      return { k, term: s.term, y: scale.y(s.per_million[i]) ?? M.top + PLOT_H }
    }).sort((a, b) => a.y - b.y)
    for (let j = 1; j < items.length; j += 1) {
      items[j].y = Math.max(items[j].y, items[j - 1].y + LABEL_GAP)
    }
    return items
  }, [shown, scale])

  function onMove(e) {
    const rect = svgRef.current.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    const i = Math.round(((px - M.left) / PLOT_W) * (decades.length - 1))
    setHover(Math.max(0, Math.min(decades.length - 1, i)))
  }

  const slotOf = (s) => series.indexOf(s) + 1

  return (
    <div className="trend-root">
      <div className="trend-plot" ref={plotRef}>
        <svg
          ref={svgRef}
          className="trend-svg"
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Popularity over time of ${shown.map((s) => s.term).join(', ')}`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {scale.ticks.map((t) => (
            <g key={t}>
              <line className="trend-grid" x1={M.left} x2={M.left + PLOT_W} y1={scale.y(t)} y2={scale.y(t)} />
              <text className="trend-axis" x={M.left - 8} y={scale.y(t)} dy="0.32em" textAnchor="end">
                {formatRate(t)}
              </text>
            </g>
          ))}
          {decades.map((d, i) => i % every === 0 && (
            <text key={d} className="trend-axis" x={x(i)} y={H - 10} textAnchor="middle">
              {d}
            </text>
          ))}
          <text className="trend-axis trend-unit" x={M.left} y={M.top - 4}>per million words</text>

          {hover !== null && (
            <line className="trend-crosshair" x1={x(hover)} x2={x(hover)} y1={M.top} y2={M.top + PLOT_H} />
          )}
          {shown.map((s, k) => (
            <path key={s.term} d={paths[k]} className={`trend-line trend-slot-${slotOf(s)}`} />
          ))}
          {hover !== null && shown.map((s) => {
            const yy = scale.y(s.per_million[hover])
            return yy === null ? null : (
              <circle key={s.term} cx={x(hover)} cy={yy} r={4.5} className={`trend-dot trend-slot-${slotOf(s)}`} />
            )
          })}
          {endLabels.map((l) => (
            <g key={l.term} transform={`translate(${M.left + PLOT_W + 6},${l.y})`}>
              <line x1={0} x2={10} y1={0} y2={0} className={`trend-line trend-slot-${slotOf(shown[l.k])}`} />
              <text className="trend-endlabel" x={14} dy="0.32em">{l.term}</text>
            </g>
          ))}
        </svg>

        {hover !== null && (
          <div
            className="trend-tooltip"
            style={{ left: `${(x(hover) / W) * 100}%`, transform: hover > decades.length / 2 ? 'translateX(calc(-100% - 12px))' : 'translateX(12px)' }}
          >
            <div className="trend-tooltip-title">{decades[hover]}s</div>
            {shown.map((s) => (
              <div key={s.term} className="trend-tooltip-row">
                <span className={`trend-swatch trend-slot-${slotOf(s)}`} />
                <span className="trend-tooltip-term">{s.term}</span>
                <span className="trend-tooltip-value">{formatRate(s.per_million[hover])}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <details className="trend-table-toggle">
        <summary>Show as a table</summary>
        <table className="trend-table">
          <thead>
            <tr>
              <th>Decade</th>
              {shown.map((s) => <th key={s.term}>{s.term}</th>)}
            </tr>
          </thead>
          <tbody>
            {decades.map((d, i) => (
              <tr key={d}>
                <td>{d}s</td>
                {shown.map((s) => <td key={s.term}>{formatRate(s.per_million[i])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  )
}

export default NgramTrendChart
