import './GrowthChart.css'

const PLOT_HEIGHT = 90 // px
const POINT_SPACING = 10 // px between two consecutive days
const PADDING_X = 16 // px -- room for the first/last point's own hit-circle and label to sit inside the viewBox rather than getting clipped by it
const LABEL_EVERY = 7 // one date label per week of points, not per point -- see below

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
  if (!data || data.length === 0) return null
  const maxValue = Math.max(...data.map((d) => d.value), 1)
  const lastIndex = data.length - 1
  const width = PADDING_X * 2 + POINT_SPACING * lastIndex

  function x(i) {
    return PADDING_X + i * POINT_SPACING
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
    <div className="growth-chart">
      <svg
        className="growth-chart-svg"
        width={width}
        height={PLOT_HEIGHT}
        viewBox={`0 0 ${width} ${PLOT_HEIGHT}`}
        preserveAspectRatio="none"
      >
        <line x1={0} y1={PLOT_HEIGHT} x2={width} y2={PLOT_HEIGHT} className="growth-chart-baseline" />
        <path d={areaPath} className="growth-chart-area" />
        <polyline points={linePoints} className="growth-chart-line" />
        {data.map((d, i) => (
          <circle key={d.date} cx={x(i)} cy={y(d.value)} r={5} className="growth-chart-hit">
            <title>{`${formatShort(d.date)}: ${d.value.toLocaleString()} ${unit}`}</title>
          </circle>
        ))}
      </svg>
      <div className="growth-chart-labels" style={{ width }}>
        {data.map((d, i) => {
          // The last point always gets a label; a periodic one this close
          // to it would just collide with that label's text, so it's
          // skipped rather than shown crowded.
          const showLabel = i === 0 || i === lastIndex || (i % LABEL_EVERY === 0 && lastIndex - i >= 3)
          if (!showLabel) return null
          return (
            <span key={d.date} className="growth-chart-label" style={{ left: x(i) }}>
              {formatShort(d.date)}
            </span>
          )
        })}
      </div>
    </div>
  )
}

export default GrowthChart
