import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { ZOOM_MS, ZOOM_PADDING, cssVar } from './graphUtils'
import SharedWordsPanel from './SharedWordsPanel'
import './GraphView.css'

const TOP_K_OPTIONS = [2, 3, 4, 5]
const LINK_DISTANCE_MIN = 40
const LINK_DISTANCE_MAX = 220
const NODE_RADIUS_CENTER = 14
const NODE_RADIUS_RELATED = 8
const RING_2_ALPHA = 0.45 // second-hop node/label fade, matching GraphView.jsx's own constant
// d3-force-3d's own forceManyBody() default strength -- the baseline the
// Spread slider scales, not a value chosen for this graph specifically.
const CHARGE_BASE = -30
const SPREAD_MIN = 0.3
const SPREAD_MAX = 5
const SPREAD_DEFAULT = 1
const ZOOM_MIN = 0.1
const ZOOM_MAX = 8
const ZOOM_DEFAULT = 1

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
function RelatednessGraph({ initialId, fetchUrl, getLabel, getSublabel, onNodeNavigate, sharedWordsUrl, highlightId, scope }) {
  const [center, setCenter] = useState(null)
  const [rawGraph, setRawGraph] = useState({ nodes: [], edges: [] })
  const [topK, setTopK] = useState(3)
  const [spread, setSpread] = useState(SPREAD_DEFAULT)
  const [zoomLevel, setZoomLevel] = useState(ZOOM_DEFAULT)
  const [comparePair, setComparePair] = useState(null) // {sourceId, targetId, sourceLabel, targetLabel} | null
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dims, setDims] = useState({ width: 0, height: 0 })
  const [isFullscreen, setIsFullscreen] = useState(false)
  const viewRef = useRef(null)
  const containerRef = useRef(null)
  const fgRef = useRef(null)
  const chaseIntervalRef = useRef(null)
  // Once the user has manually zoomed/panned (wheel, drag, pinch, or the
  // Zoom slider), every AUTOMATIC re-fit below (the chase loop, onEngineStop,
  // and resize) backs off and leaves their framing alone -- until the node
  // SET itself actually changes (a fresh loadGraph, not a Spread tweak on
  // the same data), which does warrant a fresh fit. Before this guard
  // existed, zooming in was a losing fight: the chase loop alone could
  // re-fit every 300ms for up to 15s after any load or Spread change,
  // and onEngineStop/resize could re-fit indefinitely beyond that.
  const userAdjustedRef = useRef(false)
  // Distinguishes OUR OWN zoomToFit calls from a real user zoom inside
  // onZoom (force-graph's 'zoom' event fires identically for both -- see
  // doZoomToFit below) -- true only for the brief window an automatic fit
  // is actually animating.
  const programmaticZoomRef = useRef(false)
  const programmaticZoomTimeoutRef = useRef(null)

  // Every AUTOMATIC zoom (as opposed to the user dragging the Zoom slider,
  // which is a real user choice and deliberately NOT routed through here --
  // see handleZoom) must go through this so onZoom can tell the two apart.
  const doZoomToFit = useCallback((duration, padding) => {
    programmaticZoomRef.current = true
    if (programmaticZoomTimeoutRef.current) clearTimeout(programmaticZoomTimeoutRef.current)
    programmaticZoomTimeoutRef.current = setTimeout(() => {
      programmaticZoomRef.current = false
    }, duration + 60)
    fgRef.current?.zoomToFit(duration, padding)
  }, [])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      setDims({ width, height })
      if (userAdjustedRef.current) return
      setTimeout(() => doZoomToFit(ZOOM_MS, ZOOM_PADDING), ZOOM_MS)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [doZoomToFit])

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

  // Applies the current `spread` to both the link (spring) and charge
  // (repulsion) forces and reheats -- shared by a fresh data load AND by
  // the Spread slider itself (dragging it should visibly loosen/tighten
  // the SAME node set immediately, not trigger a network refetch). Longer
  // link distance / weaker (more-negative) charge = looser; both are
  // scaled by the identical multiplier so the two forces move together
  // rather than fighting each other at the extremes.
  const applyForcesAndChaseZoom = useCallback(
    (nodeCount) => {
      // Longer link = weaker relationship: invert score (similarity,
      // higher = closer) into a d3-force distance (lower = closer), THEN
      // scale the whole curve by spread.
      fgRef.current
        ?.d3Force('link')
        ?.distance((l) => (LINK_DISTANCE_MAX - l.score * (LINK_DISTANCE_MAX - LINK_DISTANCE_MIN)) * spread)
      fgRef.current?.d3Force('charge')?.strength(CHARGE_BASE * spread)
      fgRef.current?.d3ReheatSimulation?.()
      // Once the user has taken control of the camera, Spread should still
      // visibly loosen/tighten the layout (the forces above already ran)
      // but must NOT yank the view back to a fit -- that's the exact fight
      // this whole guard exists to end. A fresh loadGraph resets
      // userAdjustedRef to false before calling this, so a genuinely new
      // node set still gets its normal auto-fit chase below.
      if (chaseIntervalRef.current) clearInterval(chaseIntervalRef.current)
      if (userAdjustedRef.current) return
      // A single zoomToFit right after reheating races the simulation:
      // ForceGraph2D mounts and starts ticking with d3-force's default
      // 30px link distance BEFORE this effect gets a chance to override
      // it, so one immediate fit locks pan/zoom to that tight, wrong-scale
      // snapshot -- then the reheated sim spends the next several seconds
      // expanding toward the real (up to LINK_DISTANCE_MAX * spread)
      // distances, drifting nodes outside the already-locked viewport
      // (visible live as edges running off the edge of the canvas with no
      // node circle in sight). Chase it instead: re-fit every 300ms for a
      // few seconds so the view tracks the layout as it actually expands,
      // rather than guessing a single delay that's either too early (mid-
      // expansion) or wastes time waiting past when it's needed.
      // onEngineStop's own zoomToFit (below) is the final correction once
      // the simulation naturally cools down.
      //
      // The budget below was tuned against the ~60-node volume-scope
      // graph -- the fame-scope Graph tab (~220 nodes, ~1,000 edges, no
      // volume-style cap) simply takes longer per tick to settle, so a
      // fixed 4s chase gave up mid-expansion and looked frozen (found
      // live: switching top-K under the Fame toggle appeared to do
      // nothing). Scale the budget with graph size instead of raising the
      // fixed constant, so small ego graphs stay snappy.
      let elapsed = 0
      const chaseBudgetMs = Math.min(4000 + nodeCount * 40, 15000)
      chaseIntervalRef.current = setInterval(() => {
        // The user can start dragging/zooming mid-chase -- stop fighting
        // them immediately rather than waiting out the remaining budget.
        if (userAdjustedRef.current) {
          clearInterval(chaseIntervalRef.current)
          return
        }
        doZoomToFit(300, ZOOM_PADDING)
        elapsed += 300
        if (elapsed > chaseBudgetMs) clearInterval(chaseIntervalRef.current)
      }, 300)
    },
    [spread, doZoomToFit],
  )

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
          // A genuinely new node set (new topK/scope/entity) deserves a
          // fresh auto-fit even if the user had taken control of the
          // PREVIOUS graph's camera -- only Spread tweaks on the SAME data
          // (handled entirely inside applyForcesAndChaseZoom) respect that.
          userAdjustedRef.current = false
          setTimeout(() => applyForcesAndChaseZoom(data.nodes.length), 50)
        })
        .catch((err) => setError(err.message || 'failed to load graph'))
        .finally(() => setLoading(false))
    },
    [fetchUrl, applyForcesAndChaseZoom],
  )

  // Dragging the Spread slider re-applies forces to the graph ALREADY on
  // screen -- no refetch, since the node/edge data itself hasn't changed,
  // only how loosely the simulation should hold it. Guarded on node count
  // so this doesn't fire before any graph has ever loaded.
  useEffect(() => {
    if (rawGraph.nodes.length > 0) applyForcesAndChaseZoom(rawGraph.nodes.length)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spread])

  // Re-fires on topK change too (not mount-only) so the top-N control
  // re-fetches rather than just re-filtering already-truncated data. scope
  // is in the same boat: it's baked into the fetchUrl closure the caller
  // passes, but fetchUrl's own identity is deliberately NOT a dependency
  // here (an inline arrow function recreated every parent render would
  // otherwise refetch constantly) -- so callers that vary scope (the
  // global Volume/Fame toggle on Authors-/BooksRelatedness) pass it
  // through explicitly just so it can sit in this array.
  useEffect(() => {
    if (initialId != null) loadGraph(initialId, topK)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialId, topK, scope])

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
  // based default. Returns a SCREEN-pixel radius (divided by globalScale),
  // same "constant regardless of zoom" treatment paintNode already gave
  // lineWidth/font size below but had never applied to the radius itself --
  // that gap is why nodes shrank to near-invisible whenever the Spread
  // slider (or the Zoom slider, or the auto zoom-to-fit reacting to either)
  // pushed globalScale down: the circle was being drawn at a FIXED size in
  // graph-space units, so it scaled down 1:1 with the zoom level the same
  // way a node's actual world-space position does, instead of staying a
  // legible fixed size on screen the way a UI affordance should.
  const nodeRadius = useCallback(
    (node, globalScale) => (node.id === center?.id ? NODE_RADIUS_CENTER : NODE_RADIUS_RELATED) / globalScale,
    [center],
  )

  const paintNode = useMemo(
    () => (node, ctx, globalScale) => {
      const isCenter = node.id === center?.id
      const isHighlighted = highlightId != null && node.id === highlightId
      // Second-hop nodes (ring 2) fade rather than disappear -- same
      // RING_2_ALPHA treatment as the word graph's own ring 2.
      const isOuter = node.ring === 2
      const r = nodeRadius(node, globalScale)
      ctx.globalAlpha = isOuter ? RING_2_ALPHA : 1
      ctx.beginPath()
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
      ctx.fillStyle = isCenter ? cssVar('--graph-center', '#e6d200') : cssVar('--accent', '#1c6dbd')
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
      ctx.fillText(getLabel(node), node.x, node.y + r + 2 / globalScale)
      ctx.globalAlpha = 1
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
  // Takes globalScale too (force-graph's shadow-canvas render loop does
  // pass it as a 4th arg despite react-force-graph-2d's own nodeCanvasObject
  // JSDoc only showing 3 -- confirmed against CanvasPointerAreaPaintFn's own
  // type, which lists all 4) so the hit-area radius matches paintNode's
  // now-zoom-compensated one exactly, not just at the zoom level this was
  // originally written against.
  const paintNodePointerArea = useMemo(
    () => (node, color, ctx, globalScale) => {
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(node.x, node.y, nodeRadius(node, globalScale), 0, 2 * Math.PI)
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

  const handleEngineStop = useCallback(() => {
    if (userAdjustedRef.current) return
    doZoomToFit(ZOOM_MS, ZOOM_PADDING)
  }, [doZoomToFit])

  // Fires for EVERY zoom-transform change, not just wheel/pinch/drag --
  // also every animated step of doZoomToFit's own transitions (data load,
  // Spread changes, resize, onEngineStop). That's what keeps the slider
  // honest: it always reflects whatever's actually on screen, regardless
  // of which of those triggered it, without this component needing to
  // track "which one last moved the camera." Clamped into the slider's
  // own range purely for the handle's visual position -- the real zoom
  // can still sit outside [ZOOM_MIN, ZOOM_MAX] (e.g. a fresh zoomToFit on
  // a very large or very small graph) and the slider just shows maxed out
  // at whichever end until the user drags it back into range.
  //
  // Also the one place that tells a real user zoom (wheel/pinch/drag, or
  // dragging the Zoom slider -- both land here, unlike doZoomToFit's
  // programmatic calls, which are excluded via programmaticZoomRef) apart
  // from an automatic re-fit: if this wasn't us, the user has taken the
  // camera, so every future auto-fit above backs off until the next fresh
  // loadGraph. Cancels the chase loop immediately too, in case the user
  // starts interacting mid-chase rather than waiting for it to notice on
  // its next tick.
  const handleZoom = useCallback((transform) => {
    setZoomLevel(Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, transform.k)))
    if (!programmaticZoomRef.current) {
      userAdjustedRef.current = true
      if (chaseIntervalRef.current) clearInterval(chaseIntervalRef.current)
    }
  }, [])

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
        <label className="graph-range-control">
          Spread
          <input
            type="range"
            min={SPREAD_MIN}
            max={SPREAD_MAX}
            step={0.1}
            value={spread}
            onChange={(e) => setSpread(Number(e.target.value))}
            aria-label="Node spacing -- lower is tighter, higher is more spread out"
          />
        </label>
        <label className="graph-range-control">
          Zoom
          <input
            type="range"
            min={ZOOM_MIN}
            max={ZOOM_MAX}
            step={0.05}
            value={zoomLevel}
            onChange={(e) => {
              const k = Number(e.target.value)
              setZoomLevel(k)
              fgRef.current?.zoom(k, 200)
            }}
            aria-label="Zoom level"
          />
        </label>
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
            // d3-force's default alpha decay (0.0228) is tuned for small
            // ego graphs -- ticks-to-cool is roughly constant regardless of
            // node count, but each tick costs more with more nodes/edges,
            // so wall-clock time to onEngineStop grows with graph size.
            // Decaying faster on the ~200+ node fame-scope graph trades a
            // bit of layout quality for actually reaching a settled state
            // (and firing onEngineStop's own zoomToFit) instead of still
            // visibly drifting when the chase loop above gives up.
            d3AlphaDecay={graphData.nodes.length > 100 ? 0.05 : 0.0228}
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
            onZoom={handleZoom}
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
