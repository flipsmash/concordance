import { useEffect, useMemo, useRef, useState } from 'react'
import { colorForCluster } from './clusterColors'
import { cssVar } from './graphUtils'
import './BookDendrogram.css'
import './GraphView.css' // .graph-maximize -- reused here for the fullscreen button

const API_BASE = ''
const WIDTH = 1000
const HEIGHT = 500
const LEAF_PADDING = 20
const TOP_PADDING = 20
const BOTTOM_PADDING = 30

// Book-level counterpart to AuthorDendrogram.jsx -- see that file for the
// full rationale (hand-rolled SVG tree, no leaf labels by default, cluster-
// colored leaves joined from /map). Leaves here carry {id, title, author}
// (see BookDendrogramNode in browse.py) rather than one string, since a
// book needs id for identity/navigation and title for display -- unlike an
// author, where the name serves both jobs.
function BookDendrogram({ onBookClick, highlightBookId, scope = 'volume' }) {
  const [tree, setTree] = useState(null)
  const [leafOrder, setLeafOrder] = useState([])
  const [clusterByBookId, setClusterByBookId] = useState({})
  const [error, setError] = useState('')
  const [hoveredBookId, setHoveredBookId] = useState(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const containerRef = useRef(null)

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

  const activeBookId = hoveredBookId ?? highlightBookId ?? null

  useEffect(() => {
    setTree(null)
    fetch(`${API_BASE}/api/browse/books/dendrogram?scope=${scope}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => {
        setTree(data.tree)
        setLeafOrder(data.leaf_order)
      })
      .catch((err) => setError(err.message || 'failed to load dendrogram'))

    fetch(`${API_BASE}/api/browse/books/map?scope=${scope}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data) return
        setClusterByBookId(Object.fromEntries(data.nodes.map((n) => [n.id, n.cluster_id])))
      })
      .catch(() => {})
  }, [scope])

  const layout = useMemo(() => {
    if (!tree || leafOrder.length === 0) return null
    const leafIndex = new Map(leafOrder.map((b, i) => [b.id, i]))
    const maxDistance = tree.distance || 1
    const xStep = leafOrder.length > 1 ? (WIDTH - 2 * LEAF_PADDING) / (leafOrder.length - 1) : 0
    const yFor = (distance) =>
      TOP_PADDING + (1 - distance / maxDistance) * (HEIGHT - TOP_PADDING - BOTTOM_PADDING)

    const edges = []
    const leaves = []

    function visit(node) {
      if (node.id != null) {
        const x = LEAF_PADDING + leafIndex.get(node.id) * xStep
        const y = HEIGHT - BOTTOM_PADDING
        leaves.push({ id: node.id, title: node.title, author: node.author, x, y })
        return { x, y }
      }
      const left = visit(node.left)
      const right = visit(node.right)
      const x = (left.x + right.x) / 2
      const y = yFor(node.distance)
      edges.push({ x1: left.x, y1: left.y, x2: left.x, y2: y })
      edges.push({ x1: left.x, y1: y, x2: right.x, y2: y })
      edges.push({ x1: right.x, y1: right.y, x2: right.x, y2: y })
      return { x, y }
    }
    visit(tree)
    return { edges, leaves }
  }, [tree, leafOrder])

  return (
    <div className="book-dendrogram" ref={containerRef}>
      <div className="graph-controls book-dendrogram-controls">
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>
      {error && <div className="graph-error">{error}</div>}
      {tree === null && !error && <div className="graph-loading">Loading…</div>}
      {tree !== null && leafOrder.length === 0 && (
        <div className="graph-empty">
          No clustering data yet — run `concordance book-clustering{scope === 'fame' ? ' --min-fame 8' : ''}`.
        </div>
      )}
      {layout && (
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="book-dendrogram-svg" role="img" aria-label="Book dendrogram">
          {layout.edges.map((e, i) => (
            <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} className="book-dendrogram-edge" />
          ))}
          {layout.leaves.map((l) => {
            const isHighlighted = l.id === highlightBookId
            const isActive = l.id === activeBookId
            return (
            <g
              key={l.id}
              onClick={() => onBookClick?.(l)}
              onMouseEnter={() => setHoveredBookId(l.id)}
              onMouseLeave={() => setHoveredBookId((h) => (h === l.id ? null : h))}
              className="book-dendrogram-leaf"
            >
              <circle
                cx={l.x}
                cy={l.y}
                r={isActive ? 5 : 3}
                fill={colorForCluster(clusterByBookId[l.id] ?? 0)}
                stroke={isHighlighted ? cssVar('--accent', '#1c6dbd') : undefined}
                strokeWidth={isHighlighted ? 2 : undefined}
              />
              {isActive && (
                <text x={l.x} y={l.y + 14} textAnchor="middle" className="book-dendrogram-label">
                  {l.title}
                  {l.author ? ` · ${l.author}` : ''}
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

export default BookDendrogram
