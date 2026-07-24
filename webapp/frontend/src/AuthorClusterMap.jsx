import { useEffect, useMemo, useRef, useState } from 'react'
import { colorForCluster } from './clusterColors'
import { cssVar } from './graphUtils'
import './AuthorClusterMap.css'
import './GraphView.css' // .graph-maximize -- reused here for the fullscreen button

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
function AuthorClusterMap({ onAuthorClick, highlightAuthor }) {
  const [nodes, setNodes] = useState(null) // null = loading
  const [error, setError] = useState('')
  const [hovered, setHovered] = useState(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
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

  useEffect(() => {
    function handleFullscreenChange() {
      setIsFullscreen(document.fullscreenElement === containerRef.current)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  function toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      containerRef.current?.requestFullscreen()
    }
  }

  // hover wins while active (lets you inspect a different point without
  // losing your highlight pick -- it reappears the moment you mouse away).
  const active = hovered ?? highlightAuthor ?? null

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
      <div className="graph-controls cluster-map-controls">
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>
      {error && <div className="graph-error">{error}</div>}
      {nodes === null && !error && <div className="graph-loading">Loading…</div>}
      {nodes !== null && nodes.length === 0 && !error && (
        <div className="graph-empty">No cluster data yet -- run `concordance author-clustering`.</div>
      )}
      {points.length > 0 && (
        <svg viewBox={`0 0 ${VIEW} ${VIEW}`} className="cluster-map-svg" role="img" aria-label="Author cluster map">
          {points.map((p) => {
            const isHighlighted = p.author === highlightAuthor
            const isActive = p.author === active
            return (
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
                  opacity={active === null || isActive ? 1 : 0.35}
                  stroke={isHighlighted ? cssVar('--accent', '#aa3bff') : undefined}
                  strokeWidth={isHighlighted ? 3 : undefined}
                />
                {isActive && (
                  <text x={p.cx} y={p.cy - radiusFor(p.book_count) - 4} textAnchor="middle" className="cluster-map-label">
                    {p.author} · {p.book_count} book{p.book_count === 1 ? '' : 's'}
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}

export default AuthorClusterMap
