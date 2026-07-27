import { useEffect, useMemo, useRef, useState } from 'react'
import { colorForCluster } from './clusterColors'
import { cssVar } from './graphUtils'
import './BookClusterMap.css'
import './GraphView.css' // .graph-maximize -- reused here for the fullscreen button

const API_BASE = ''
const VIEW = 600 // SVG viewBox is VIEW x VIEW, coordinates normalized into it
const PADDING = 40
const RADIUS_MIN = 4
const RADIUS_MAX = 14

// Book-level counterpart to AuthorClusterMap.jsx -- see that file for the
// full rationale (classical MDS position, cluster color, plain SVG since
// these points are precomputed/static). Highlighted/clicked by book id,
// not title (unlike an author, a title alone isn't guaranteed unique).
function BookClusterMap({ onBookClick, highlightBookId }) {
  const [nodes, setNodes] = useState(null) // null = loading
  const [error, setError] = useState('')
  const [hovered, setHovered] = useState(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const containerRef = useRef(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/books/map`)
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
  const active = hovered ?? highlightBookId ?? null

  const { points, maxWordCount } = useMemo(() => {
    if (!nodes || nodes.length === 0) return { points: [], maxWordCount: 1 }
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
    const maxWords = Math.max(...nodes.map((n) => n.word_count), 1)
    return {
      maxWordCount: maxWords,
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

  function radiusFor(wordCount) {
    const t = Math.sqrt(wordCount / maxWordCount) // area-proportional, not radius-proportional
    return RADIUS_MIN + t * (RADIUS_MAX - RADIUS_MIN)
  }

  return (
    <div className="book-cluster-map" ref={containerRef}>
      <div className="graph-controls book-cluster-map-controls">
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>
      {error && <div className="graph-error">{error}</div>}
      {nodes === null && !error && <div className="graph-loading">Loading…</div>}
      {nodes !== null && nodes.length === 0 && !error && (
        <div className="graph-empty">No cluster data yet -- run `concordance book-clustering`.</div>
      )}
      {points.length > 0 && (
        <svg viewBox={`0 0 ${VIEW} ${VIEW}`} className="book-cluster-map-svg" role="img" aria-label="Book cluster map">
          {points.map((p) => {
            const isHighlighted = p.id === highlightBookId
            const isActive = p.id === active
            return (
              <g
                key={p.id}
                className="book-cluster-map-point"
                onClick={() => onBookClick?.(p)}
                onMouseEnter={() => setHovered(p.id)}
                onMouseLeave={() => setHovered((h) => (h === p.id ? null : h))}
              >
                <circle
                  cx={p.cx}
                  cy={p.cy}
                  r={radiusFor(p.word_count)}
                  fill={colorForCluster(p.cluster_id)}
                  opacity={active === null || isActive ? 1 : 0.35}
                  stroke={isHighlighted ? cssVar('--accent', '#1c6dbd') : undefined}
                  strokeWidth={isHighlighted ? 3 : undefined}
                />
                {isActive && (
                  <text x={p.cx} y={p.cy - radiusFor(p.word_count) - 4} textAnchor="middle" className="book-cluster-map-label">
                    {p.title}
                    {p.author ? ` · ${p.author}` : ''} · {p.word_count} {p.word_count === 1 ? 'word' : 'words'}
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

export default BookClusterMap
