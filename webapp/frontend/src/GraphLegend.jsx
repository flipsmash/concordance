import { useEffect, useState } from 'react'
import { colorForBucket, UNCATEGORIZED_GRAY } from './domainColors'

const API_BASE = ''

// Fetched independently of any search, so the legend is visible immediately —
// color is domain identity, and identity should never be color-alone, so the
// full legend is shown regardless of which buckets happen to appear in the
// current graph.
function GraphLegend() {
  const [entries, setEntries] = useState([])

  useEffect(() => {
    fetch(`${API_BASE}/api/graph/legend`)
      .then((res) => res.json())
      .then(setEntries)
      .catch(() => {})
  }, [])

  return (
    <div className="graph-legend">
      {entries.map((e) => (
        <span className="graph-legend-item" key={e.bucket}>
          <span className="graph-legend-swatch" style={{ background: colorForBucket(e.bucket) }} />
          {e.name}
        </span>
      ))}
      <span className="graph-legend-item">
        <span className="graph-legend-swatch" style={{ background: UNCATEGORIZED_GRAY }} />
        Uncategorized
      </span>
    </div>
  )
}

export default GraphLegend
