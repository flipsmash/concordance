import { useState } from 'react'
import { cssVar } from './graphUtils'

/** Horizontal single-hue bar chart -- reused for "accuracy by question type"
 * and "accuracy by domain bucket." One hue throughout: the categories are
 * already distinguished by their axis labels, so a second hue per bar would
 * be redundant re-encoding, not identity (see the dataviz skill's form
 * rules). A bucket with zero attempts still renders as an explicit
 * zero-length bar with a "no data yet" tooltip -- never omitted, so the bar
 * count never silently drops below what the caller passed in. */
function SingleHueBarChart({ buckets }) {
  const [hoverKey, setHoverKey] = useState(null)
  const accent = cssVar('--accent', '#aa3bff')

  return (
    <div className="single-hue-bar-chart">
      {buckets.map((b) => {
        const pct = b.accuracy_pct ?? 0
        const hasData = b.total > 0
        return (
          <div
            key={b.key}
            className="single-hue-bar-row"
            onMouseEnter={() => setHoverKey(b.key)}
            onMouseLeave={() => setHoverKey((k) => (k === b.key ? null : k))}
          >
            <span className="single-hue-bar-label">{b.label}</span>
            <div className="single-hue-bar-track">
              <div
                className="single-hue-bar-fill"
                style={{ width: `${pct}%`, background: hasData ? accent : 'transparent' }}
              />
            </div>
            <span className="single-hue-bar-value">{hasData ? `${pct.toFixed(0)}%` : '—'}</span>
            {hoverKey === b.key && (
              <div className="single-hue-bar-tooltip">
                {hasData
                  ? `${b.correct.toFixed(1)} / ${b.total} correct`
                  : 'No data yet'}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export default SingleHueBarChart
