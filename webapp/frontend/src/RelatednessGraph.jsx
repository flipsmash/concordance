import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { ZOOM_MS, ZOOM_PADDING, cssVar } from './graphUtils'
import SharedWordsPanel from './SharedWordsPanel'
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
function RelatednessGraph({ initialId, fetchUrl, getLabel, getSublabel, onNodeNavigate, sharedWordsUrl, highlightId }) {
  const [center, setCenter] = useState(null)
  const [rawGraph, setRawGraph] = useState({ nodes: [], edges: [] })
  const [topK, setTopK] = useState(8)
  const [comparePair, setComparePair] = useState(null) // {sourceId, targetId, sourceLabel, targetLabel} | null
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dims, setDims] = useState({ width: 0, height: 0 })
  const [isFullscreen, setIsFullscreen] = useState(false)
  const viewRef = useRef(null)
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
            // A single zoomToFit right after reheating races the simulation:
            // ForceGraph2D mounts and starts ticking with d3-force's default
            // 30px link distance BEFORE this effect gets a chance to
            // override it, so one immediate fit locks pan/zoom to that
            // tight, wrong-scale snapshot -- then the reheated sim spends
            // the next several seconds expanding toward the real (up to
            // LINK_DISTANCE_MAX) distances, drifting nodes outside the
            // already-locked viewport (visible live as edges running off
            // the edge of the canvas with no node circle in sight). Chase
            // it instead: re-fit every 300ms for a few seconds so the view
            // tracks the layout as it actually expands, rather than
            // guessing a single delay that's either too early (mid-
            // expansion) or wastes time waiting past when it's needed.
            // onEngineStop's own zoomToFit (below) is the final correction
            // once the simulation naturally cools down.
            if (chaseIntervalRef.current) clearInterval(chaseIntervalRef.current)
            let elapsed = 0
            chaseIntervalRef.current = setInterval(() => {
              fgRef.current?.zoomToFit(300, ZOOM_PADDING)
              elapsed += 300
              if (elapsed > 4000) clearInterval(chaseIntervalRef.current)
            }, 300)
          }, 50)
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
      links: rawGraph.edges.map((e) => ({
        source: e.source,
        target: e.target,
        score: e.score,
        shared_word_count: e.shared_word_count,
        is_center_edge: e.is_center_edge,
      })),
    }),
    [rawGraph],
  )

  const linkLabel = useCallback(
    (l) => `${l.shared_word_count} shared words · ${(l.score * 100).toFixed(0)}% overlap`,
    [],
  )

  // react-force-graph mutates link.source/target from a raw id into the
  // resolved node object once the simulation ticks -- handle either shape
  // rather than assuming which one a click lands on.
  function endpointId(endpoint) {
    return endpoint && typeof endpoint === 'object' ? endpoint.id : endpoint
  }

  function handleLinkClick(link) {
    if (!sharedWordsUrl) return
    const sourceId = endpointId(link.source)
    const targetId = endpointId(link.target)
    const sourceNode = graphData.nodes.find((n) => n.id === sourceId)
    const targetNode = graphData.nodes.find((n) => n.id === targetId)
    setComparePair({
      sourceId,
      targetId,
      sourceLabel: sourceNode ? getLabel(sourceNode) : String(sourceId),
      targetLabel: targetNode ? getLabel(targetNode) : String(targetId),
    })
  }

  // Shared by both paintNode (visible canvas) and paintNodePointerArea
  // (force-graph's separate hit-test canvas, see below) so the clickable
  // area always matches what's drawn, not force-graph's small nodeRelSize-
  // based default.
  const nodeRadius = useCallback((node) => (node.id === center?.id ? NODE_RADIUS_CENTER : NODE_RADIUS_RELATED), [center])

  const paintNode = useMemo(
    () => (node, ctx, globalScale) => {
      const isCenter = node.id === center?.id
      const isHighlighted = highlightId != null && node.id === highlightId
      const r = nodeRadius(node)
      ctx.beginPath()
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
      ctx.fillStyle = cssVar('--accent', '#aa3bff')
      ctx.fill()
      if (isCenter) {
        ctx.lineWidth = 2 / globalScale
        ctx.strokeStyle = cssVar('--text-h', '#08060d')
        ctx.stroke()
      }
      if (isHighlighted) {
        // A dashed OUTER ring, not a stroke on the node itself -- stays
        // visually distinct from isCenter's solid stroke so "this is the
        // author I searched for" never reads as "this is the ego center."
        ctx.beginPath()
        ctx.arc(node.x, node.y, r + 5 / globalScale, 0, 2 * Math.PI)
        ctx.setLineDash([4 / globalScale, 3 / globalScale])
        ctx.lineWidth = 2 / globalScale
        ctx.strokeStyle = cssVar('--text-h', '#08060d')
        ctx.stroke()
        ctx.setLineDash([])
      }
      const fontSize = Math.max(11 / globalScale, 3)
      ctx.font = `${fontSize}px sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillStyle = cssVar('--text-h', '#08060d')
      ctx.fillText(getLabel(node), node.x, node.y + r + 2)
    },
    [center, getLabel, nodeRadius, highlightId],
  )

  // force-graph hit-tests clicks against a separate, hidden "shadow" canvas
  // that it renders itself -- nodeCanvasObject (paintNode, above) is ONLY
  // ever applied to the visible canvas (confirmed in force-graph's source:
  // nodeCanvasObject is bound to state.forceGraph alone, not
  // state.shadowGraph), so without this, the shadow canvas falls back to
  // its own default hit-circle sized from nodeRelSize (~4px radius) instead
  // of the 8-14px circles actually drawn -- clicks anywhere but the exact
  // center pixel silently miss. Found live: every attempted node click
  // failed to fire onNodeClick at all, even dead-on-visually inside a node.
  const paintNodePointerArea = useMemo(
    () => (node, color, ctx) => {
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(node.x, node.y, nodeRadius(node), 0, 2 * Math.PI)
      ctx.fill()
    },
    [nodeRadius],
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
            nodePointerAreaPaint={paintNodePointerArea}
            linkColor={() => cssVar('--border', '#e5e4e7')}
            linkWidth={(l) => 0.5 + l.score * 2.5}
            linkLineDash={(l) => (l.is_center_edge ? null : [4, 3])}
            linkLabel={linkLabel}
            onNodeClick={handleNodeClick}
            onLinkClick={handleLinkClick}
            onEngineStop={handleEngineStop}
          />
        )}
        {graphData.nodes.length === 0 && !loading && !error && (
          <div className="graph-empty">Not enough shared vocabulary yet to show relationships.</div>
        )}
      </div>

      {comparePair && sharedWordsUrl && (
        <SharedWordsPanel
          fetchUrl={sharedWordsUrl(comparePair.sourceId, comparePair.targetId)}
          titleA={comparePair.sourceLabel}
          titleB={comparePair.targetLabel}
          onClose={() => setComparePair(null)}
        />
      )}
    </div>
  )
}

export default RelatednessGraph
