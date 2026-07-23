import { useEffect, useMemo, useRef, useState } from 'react'
import { colorForCluster } from './clusterColors'
import './AuthorClusterMap.css'

const API_BASE = ''
const VIEW = 600 // SVG viewBox is VIEW x VIEW, coordinates normalized into it
const PADDING = 40
const RADIUS_MIN = 4
const RADIUS_MAX = 14

// The cluster map: authors positioned by classical MDS (real 2D distances
// derived from the same cosine-similarity structure author_similarity's
// scores use) and colored by hierarchical cluster -- see
// compute_author_clustering in concordance/db.py. Deliberately plain SVG,
// not react-force-graph-2d/canvas: these points are precomputed and
// static (no simulation to run), so SVG's native onClick/hover avoids
// re-deriving RelatednessGraph's custom canvas hit-testing machinery for
// a case that doesn't need it.
function AuthorClusterMap({ onAuthorClick }) {
  const [nodes, setNodes] = useState(null) // null = loading
  const [error, setError] = useState('')
  const [hovered, setHovered] = useState(null)
  const containerRef = useRef(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/authors/map`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => setNodes(data.nodes))
      .catch((err) => setError(err.message || 'failed to load cluster map'))
  }, [])

  const { points, maxBookCount } = useMemo(() => {
    if (!nodes || nodes.length === 0) return { points: [], maxBookCount: 1 }
    const xs = nodes.map((n) => n.x)
    const ys = nodes.map((n) => n.y)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const spanX = maxX - minX || 1
    const spanY = maxY - minY || 1
    const span = Math.max(spanX, spanY)
    const scale = (VIEW - 2 * PADDING) / span
    const maxBooks = Math.max(...nodes.map((n) => n.book_count), 1)
    return {
      maxBookCount: maxBooks,
      points: nodes.map((n) => ({
        ...n,
        // Center the (possibly non-square) data extent within the square
        // viewBox rather than stretching x/y independently -- an MDS map's
        // aspect ratio carries meaning (distances are only comparable if
        // both axes share one scale), unlike a bar chart's independent axes.
        cx: PADDING + (n.x - minX) * scale + (VIEW - 2 * PADDING - spanX * scale) / 2,
        cy: PADDING + (maxY - n.y) * scale + (VIEW - 2 * PADDING - spanY * scale) / 2,
      })),
    }
  }, [nodes])

  function radiusFor(bookCount) {
    const t = Math.sqrt(bookCount / maxBookCount) // area-proportional, not radius-proportional
    return RADIUS_MIN + t * (RADIUS_MAX - RADIUS_MIN)
  }

  return (
    <div className="cluster-map" ref={containerRef}>
      {error && <div className="graph-error">{error}</div>}
      {nodes === null && !error && <div className="graph-loading">Loading…</div>}
      {nodes !== null && nodes.length === 0 && !error && (
        <div className="graph-empty">No cluster data yet -- run `concordance author-clustering`.</div>
      )}
      {points.length > 0 && (
        <svg viewBox={`0 0 ${VIEW} ${VIEW}`} className="cluster-map-svg" role="img" aria-label="Author cluster map">
          {points.map((p) => (
            <g
              key={p.author}
              className="cluster-map-point"
              onClick={() => onAuthorClick?.(p.author)}
              onMouseEnter={() => setHovered(p.author)}
              onMouseLeave={() => setHovered((h) => (h === p.author ? null : h))}
            >
              <circle
                cx={p.cx}
                cy={p.cy}
                r={radiusFor(p.book_count)}
                fill={colorForCluster(p.cluster_id)}
                opacity={hovered === null || hovered === p.author ? 1 : 0.35}
              />
              {hovered === p.author && (
                <text x={p.cx} y={p.cy - radiusFor(p.book_count) - 4} textAnchor="middle" className="cluster-map-label">
                  {p.author} · {p.book_count} book{p.book_count === 1 ? '' : 's'}
                </text>
              )}
            </g>
          ))}
        </svg>
      )}
    </div>
  )
}

export default AuthorClusterMap
