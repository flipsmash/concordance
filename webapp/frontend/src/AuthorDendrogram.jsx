import { useEffect, useMemo, useState } from 'react'
import { colorForCluster } from './clusterColors'
import './AuthorDendrogram.css'

const API_BASE = ''
const WIDTH = 1000
const HEIGHT = 500
const LEAF_PADDING = 20
const TOP_PADDING = 20
const BOTTOM_PADDING = 30

// Hand-rolled SVG tree (x from leaf order, y from merge distance) -- no
// charting library, matching this project's established no-D3 convention
// (GraphView.jsx, RelatednessGraph.jsx, AuthorClusterMap.jsx all roll
// their own rendering too). Leaf text labels are deliberately omitted by
// default -- 200 of them crammed along one axis would be illegible --
// hover reveals identity instead, mirroring AuthorClusterMap's own
// hover-then-label pattern. Leaf dots are colored by cluster (fetched from
// /map and joined by author name) so this view visually agrees with the
// Map tab instead of introducing its own unrelated color language.
function AuthorDendrogram({ onAuthorClick }) {
  const [tree, setTree] = useState(null)
  const [leafOrder, setLeafOrder] = useState([])
  const [clusterByAuthor, setClusterByAuthor] = useState({})
  const [error, setError] = useState('')
  const [hoveredAuthor, setHoveredAuthor] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/authors/dendrogram`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => {
        setTree(data.tree)
        setLeafOrder(data.leaf_order)
      })
      .catch((err) => setError(err.message || 'failed to load dendrogram'))

    fetch(`${API_BASE}/api/browse/authors/map`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data) return
        setClusterByAuthor(Object.fromEntries(data.nodes.map((n) => [n.author, n.cluster_id])))
      })
      .catch(() => {})
  }, [])

  const layout = useMemo(() => {
    if (!tree || leafOrder.length === 0) return null
    const leafIndex = new Map(leafOrder.map((a, i) => [a, i]))
    const maxDistance = tree.distance || 1
    const xStep = leafOrder.length > 1 ? (WIDTH - 2 * LEAF_PADDING) / (leafOrder.length - 1) : 0
    const yFor = (distance) =>
      TOP_PADDING + (1 - distance / maxDistance) * (HEIGHT - TOP_PADDING - BOTTOM_PADDING)

    const edges = []
    const leaves = []

    function visit(node) {
      if (node.author != null) {
        const x = LEAF_PADDING + leafIndex.get(node.author) * xStep
        const y = HEIGHT - BOTTOM_PADDING
        leaves.push({ author: node.author, x, y })
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
    <div className="author-dendrogram">
      {error && <div className="graph-error">{error}</div>}
      {tree === null && !error && <div className="graph-loading">Loading…</div>}
      {tree !== null && leafOrder.length === 0 && (
        <div className="graph-empty">No clustering data yet — run `concordance author-clustering`.</div>
      )}
      {layout && (
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="author-dendrogram-svg" role="img" aria-label="Author dendrogram">
          {layout.edges.map((e, i) => (
            <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} className="author-dendrogram-edge" />
          ))}
          {layout.leaves.map((l) => (
            <g
              key={l.author}
              onClick={() => onAuthorClick?.(l.author)}
              onMouseEnter={() => setHoveredAuthor(l.author)}
              onMouseLeave={() => setHoveredAuthor((h) => (h === l.author ? null : h))}
              className="author-dendrogram-leaf"
            >
              <circle
                cx={l.x}
                cy={l.y}
                r={hoveredAuthor === l.author ? 5 : 3}
                fill={colorForCluster(clusterByAuthor[l.author] ?? 0)}
              />
              {hoveredAuthor === l.author && (
                <text x={l.x} y={l.y + 14} textAnchor="middle" className="author-dendrogram-label">
                  {l.author}
                </text>
              )}
            </g>
          ))}
        </svg>
      )}
    </div>
  )
}

export default AuthorDendrogram
