import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { useNavigate } from 'react-router-dom'
import { cssVar, ZOOM_MS, ZOOM_PADDING } from './graphUtils'
import './GraphView.css'
import './Categories.css'

const API_BASE = ''
const NODE_RADIUS_MIN = 6
const NODE_RADIUS_MAX = 28
const LINK_WIDTH_MIN = 1
const LINK_WIDTH_MAX = 6
const LINK_DISTANCE = 2 * NODE_RADIUS_MAX + 60

function radiusForCount(count, maxCount) {
  if (maxCount <= 0) return NODE_RADIUS_MIN
  const t = Math.sqrt(count / maxCount)
  return NODE_RADIUS_MIN + t * (NODE_RADIUS_MAX - NODE_RADIUS_MIN)
}

function endpointId(endpoint) {
  return endpoint && typeof endpoint === 'object' ? endpoint.id : endpoint
}

// Copy-adapted from CategoryOverlapGraph.jsx, not a shared import -- same
// reasoning that component's own docstring gives: genre reaches a word
// through a genuinely different join (word_book/book_genre, not
// word_category directly), and genre has no drilldown tiers or bucket
// colors to carry, so forcing one abstraction over both would cost more
// than it saves. Unlike categories, genre is always the FULL flat set --
// concordance/genre.py's GENRE_LIST has no hierarchy -- so there's no
// bucket/parent prop here, just the one graph.
function GenreOverlapGraph() {
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [dims, setDims] = useState({ width: 0, height: 0 })
  const containerRef = useRef(null)
  const fgRef = useRef(null)
  const chaseIntervalRef = useRef(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      setDims({ width, height })
      setTimeout(() => fgRef.current?.zoomToFit(ZOOM_MS, ZOOM_PADDING), ZOOM_MS)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/genre-overlap`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then(setData)
      .catch((err) => setError(err.message || 'failed to load overlap'))
  }, [])

  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] }
    return {
      nodes: data.sizes.map((s) => ({ id: s.genre, name: s.genre, word_count: s.word_count })),
      links: data.cells.map((c) => ({
        source: c.genre_a, target: c.genre_b, ratio: c.ratio, shared_words: c.shared_words,
      })),
    }
  }, [data])

  const maxCount = Math.max(1, ...graphData.nodes.map((n) => n.word_count))
  const maxRatio = Math.max(0, ...graphData.links.map((l) => l.ratio))

  // See CategoryOverlapGraph.jsx's identical effect for why this chases
  // zoomToFit for a few seconds instead of firing it once: the boosted
  // charge/link-distance forces below keep expanding the layout well past
  // the first tick, so one immediate fit locks onto an early, still-
  // mid-expansion snapshot.
  useEffect(() => {
    if (graphData.nodes.length === 0) return
    const timer = setTimeout(() => {
      fgRef.current?.d3Force('link')?.distance(LINK_DISTANCE)
      fgRef.current?.d3Force('charge')?.strength(-220)
      fgRef.current?.d3ReheatSimulation?.()

      if (chaseIntervalRef.current) clearInterval(chaseIntervalRef.current)
      let elapsed = 0
      chaseIntervalRef.current = setInterval(() => {
        fgRef.current?.zoomToFit(300, ZOOM_PADDING)
        elapsed += 300
        if (elapsed > 3000) clearInterval(chaseIntervalRef.current)
      }, 300)
    }, 50)
    return () => {
      clearTimeout(timer)
      if (chaseIntervalRef.current) clearInterval(chaseIntervalRef.current)
    }
  }, [graphData])

  const nodeRadius = useCallback((node) => radiusForCount(node.word_count, maxCount), [maxCount])

  const paintNode = useCallback(
    (node, ctx, globalScale) => {
      const r = nodeRadius(node)
      ctx.beginPath()
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
      ctx.fillStyle = cssVar('--accent', '#1c6dbd')
      ctx.fill()
      ctx.lineWidth = 1.5 / globalScale
      ctx.strokeStyle = cssVar('--bg', '#fff')
      ctx.stroke()
      const fontSize = Math.max(11 / globalScale, 3)
      ctx.font = `${fontSize}px sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillStyle = cssVar('--text-h', '#08060d')
      ctx.fillText(node.name, node.x, node.y + r + 3)
    },
    [nodeRadius],
  )

  const paintNodePointerArea = useCallback(
    (node, color, ctx) => {
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(node.x, node.y, nodeRadius(node), 0, 2 * Math.PI)
      ctx.fill()
    },
    [nodeRadius],
  )

  // A genre node -> the Books page filtered to it (reuses the genre picker
  // already on that page); an edge -> the word list carrying BOTH genres
  // via all_genre (an intersection, matching exactly what this graph's own
  // shared_words count computes -- see _build_word_filters' docstring for
  // why that's a distinct param from browse_books's own OR-semantics `genre`).
  function handleNodeClick(node) {
    navigate(`/app/books?genre=${encodeURIComponent(node.id)}`)
  }

  function handleLinkClick(link) {
    const params = new URLSearchParams()
    params.append('all_genre', endpointId(link.source))
    params.append('all_genre', endpointId(link.target))
    navigate(`/app/words?${params}`)
  }

  const linkLabel = useCallback(
    (l) => `${l.shared_words.toLocaleString()} shared words (${Math.round(l.ratio * 100)}%)`,
    [],
  )
  const linkWidth = useCallback(
    (l) => LINK_WIDTH_MIN + (maxRatio > 0 ? l.ratio / maxRatio : 0) * (LINK_WIDTH_MAX - LINK_WIDTH_MIN),
    [maxRatio],
  )
  const handleEngineStop = useCallback(() => fgRef.current?.zoomToFit(ZOOM_MS, ZOOM_PADDING), [])

  const showEmpty = data && data.sizes.length < 2

  return (
    <div className="graph-view">
      {/* Always mounted -- see CategoryOverlapGraph.jsx's identical comment:
          the ResizeObserver effect needs containerRef.current on its first
          (and only) run, which an early return gated on `data` would defeat. */}
      <div className="category-overlap-canvas-wrap" ref={containerRef}>
        {error && <div className="error-banner">{error}</div>}
        {!error && !data && <div className="page-loading">Loading…</div>}
        {!error && showEmpty && (
          <p className="category-empty">Not enough genres tagged yet to compare overlap.</p>
        )}
        {!error && data && !showEmpty && (
          <ForceGraph2D
            ref={fgRef}
            width={dims.width || undefined}
            height={dims.height || undefined}
            graphData={graphData}
            nodeLabel={(n) => `<b>${n.name}</b><br/>${n.word_count.toLocaleString()} words`}
            nodeCanvasObject={paintNode}
            nodeCanvasObjectMode={() => 'replace'}
            nodePointerAreaPaint={paintNodePointerArea}
            linkColor={() => cssVar('--accent', '#1c6dbd')}
            linkWidth={linkWidth}
            linkLabel={linkLabel}
            onNodeClick={handleNodeClick}
            onLinkClick={handleLinkClick}
            onEngineStop={handleEngineStop}
            d3AlphaDecay={0.05}
            cooldownTicks={150}
          />
        )}
      </div>
      {!error && data && !showEmpty && (
        <p className="category-empty">
          Circle size = word count · line thickness = overlap strength. Click a circle to browse that genre's
          books, a line to browse words shared by both genres.
        </p>
      )}
    </div>
  )
}

export default GenreOverlapGraph
