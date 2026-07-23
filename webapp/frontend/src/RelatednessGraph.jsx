import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { ZOOM_MS, ZOOM_PADDING, cssVar } from './graphUtils'
import './GraphView.css'

const TOP_K_OPTIONS = [5, 8, 12, 20]
const LINK_DISTANCE_MIN = 40
const LINK_DISTANCE_MAX = 220
const NODE_RADIUS_CENTER = 14
const NODE_RADIUS_RELATED = 8

// Lexical-overlap relatedness graph (books/authors) -- copy-adapted from
// GraphView.jsx, not imported from it: that component isn't factored into a
// reusable core, and its word-specific concerns (domain-bucket coloring,
// zipf-based node size, the Meaning/Spelling embedding-signal toggle, the 3D
// mode) don't apply here. This is deliberately leaner: a single lexical
// shared-vocabulary metric, uniform node styling (center vs. related), 2D
// only. Always ego-anchored on one center node (never a full corpus-wide
// layout), matching the plan's scale reasoning against an all-books or
// all-authors graph on every page.
//
// score is similarity (higher = more related) -- the opposite convention
// from the word graph's `distance` (lower = closer). d3-force's link force
// wants a *distance*, so it's inverted explicitly below rather than reusing
// GraphView's assumption that the API's own number is already the right
// one to hand the simulation.
function RelatednessGraph({ initialId, fetchUrl, getLabel, getSublabel, onNodeNavigate }) {
  const [center, setCenter] = useState(null)
  const [rawGraph, setRawGraph] = useState({ nodes: [], edges: [] })
  const [topK, setTopK] = useState(8)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dims, setDims] = useState({ width: 0, height: 0 })
  const [isFullscreen, setIsFullscreen] = useState(false)
  const viewRef = useRef(null)
  const containerRef = useRef(null)
  const fgRef = useRef(null)

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
    function handleFullscreenChange() {
      setIsFullscreen(document.fullscreenElement === viewRef.current)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  function toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      viewRef.current?.requestFullscreen()
    }
  }

  const loadGraph = useCallback(
    (id, k) => {
      setLoading(true)
      setError('')
      fetch(fetchUrl(id, k))
        .then((res) => {
          if (res.status === 404) throw new Error('not found')
          if (!res.ok) throw new Error(`request failed (${res.status})`)
          return res.json()
        })
        .then((data) => {
          setCenter(data.center)
          setRawGraph({ nodes: data.nodes, edges: data.edges })
          setTimeout(() => {
            // Longer link = weaker relationship: invert score (similarity,
            // higher = closer) into a d3-force distance (lower = closer).
            fgRef.current
              ?.d3Force('link')
              ?.distance((l) => LINK_DISTANCE_MAX - l.score * (LINK_DISTANCE_MAX - LINK_DISTANCE_MIN))
            fgRef.current?.d3ReheatSimulation?.()
            fgRef.current?.zoomToFit(ZOOM_MS, ZOOM_PADDING)
          }, ZOOM_MS)
        })
        .catch((err) => setError(err.message || 'failed to load graph'))
        .finally(() => setLoading(false))
    },
    [fetchUrl],
  )

  // Re-fires on topK change too (not mount-only) so the top-N control
  // re-fetches rather than just re-filtering already-truncated data.
  useEffect(() => {
    if (initialId != null) loadGraph(initialId, topK)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialId, topK])

  function handleNodeClick(node) {
    if (node.id === center?.id) return
    onNodeNavigate?.(node)
  }

  const graphData = useMemo(
    () => ({
      nodes: rawGraph.nodes.map((n) => ({ ...n })),
      links: rawGraph.edges.map((e) => ({ source: e.source, target: e.target, score: e.score })),
    }),
    [rawGraph],
  )

  const paintNode = useMemo(
    () => (node, ctx, globalScale) => {
      const isCenter = node.id === center?.id
      const r = isCenter ? NODE_RADIUS_CENTER : NODE_RADIUS_RELATED
      ctx.globalAlpha = isCenter ? 1 : 0.6
      ctx.beginPath()
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
      ctx.fillStyle = cssVar('--accent', '#aa3bff')
      ctx.fill()
      ctx.globalAlpha = 1
      if (isCenter) {
        ctx.lineWidth = 2 / globalScale
        ctx.strokeStyle = cssVar('--text-h', '#08060d')
        ctx.stroke()
      }
      const fontSize = Math.max(11 / globalScale, 3)
      ctx.font = `${fontSize}px sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillStyle = cssVar('--text-h', '#08060d')
      ctx.fillText(getLabel(node), node.x, node.y + r + 2)
    },
    [center, getLabel],
  )

  const nodeLabel = useCallback(
    (n) => {
      const sub = getSublabel?.(n)
      return `<b>${getLabel(n)}</b>${sub ? `<br/>${sub}` : ''}`
    },
    [getLabel, getSublabel],
  )

  const handleEngineStop = useCallback(() => fgRef.current?.zoomToFit(ZOOM_MS, ZOOM_PADDING), [])

  return (
    <div className="graph-view" ref={viewRef}>
      <div className="graph-controls">
        <div className="graph-signal-toggle" role="group" aria-label="Number of related items shown">
          {TOP_K_OPTIONS.map((k) => (
            <button key={k} type="button" className={topK === k ? 'active' : ''} onClick={() => setTopK(k)}>
              Top {k}
            </button>
          ))}
        </div>
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>

      {error && <div className="graph-error">{error}</div>}
      {loading && <div className="graph-loading">Loading…</div>}

      <div className="graph-canvas-wrap" ref={containerRef}>
        {graphData.nodes.length > 0 && (
          <ForceGraph2D
            ref={fgRef}
            width={dims.width || undefined}
            height={dims.height || undefined}
            graphData={graphData}
            nodeLabel={nodeLabel}
            nodeCanvasObject={paintNode}
            nodeCanvasObjectMode={() => 'replace'}
            linkColor={() => cssVar('--border', '#e5e4e7')}
            linkWidth={(l) => 0.5 + l.score * 2.5}
            onNodeClick={handleNodeClick}
            onEngineStop={handleEngineStop}
          />
        )}
        {graphData.nodes.length === 0 && !loading && !error && (
          <div className="graph-empty">Not enough shared vocabulary yet to show relationships.</div>
        )}
      </div>
    </div>
  )
}

export default RelatednessGraph
