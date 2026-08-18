import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { useNavigate } from 'react-router-dom'
import { colorForBucket } from './domainColors'
import { cssVar, ZOOM_MS, ZOOM_PADDING } from './graphUtils'
import './GraphView.css'
import './Categories.css'

const API_BASE = ''
const NODE_RADIUS_MIN = 6
const NODE_RADIUS_MAX = 28
const LINK_WIDTH_MIN = 1
const LINK_WIDTH_MAX = 6
// Sized for the LARGEST possible circle pair (2*NODE_RADIUS_MAX + padding),
// not score-scaled like RelatednessGraph's own link force -- this graph's
// nodes vary hugely in radius (word count), so a distance tuned for the
// biggest pair keeps every pair clear.
const LINK_DISTANCE = 2 * NODE_RADIUS_MAX + 60

// sqrt, not linear: word counts at any drilldown level span a wide
// multiplicative range (e.g. 17 to 17,681 siblings in the same graph) --
// linear scaling would make most nodes an invisible dot next to one giant
// circle, the same reasoning graphUtils.radiusForZipf uses for word nodes
// (not reused directly: that one's keyed to a fixed 1-7 zipf scale, this is
// relative to whatever the current graph's own max happens to be).
function radiusForCount(count, maxCount) {
  if (maxCount <= 0) return NODE_RADIUS_MIN
  const t = Math.sqrt(count / maxCount)
  return NODE_RADIUS_MIN + t * (NODE_RADIUS_MAX - NODE_RADIUS_MIN)
}

// react-force-graph mutates link.source/target from a raw id string into
// the resolved node object once the simulation ticks -- same gotcha
// RelatednessGraph.jsx's own endpointId helper exists for.
function endpointId(endpoint) {
  return endpoint && typeof endpoint === 'object' ? endpoint.id : endpoint
}

// Force-directed replacement for the original flat heatmap (see git log --
// Brian's read on the matrix: correct but not compelling, and the "click a
// cell" affordance was easy to miss). Nodes are this drilldown level's own
// siblings (word count -> circle size), edges are word-set overlap (ratio
// -> line thickness, source/target -> the two categories) -- reuses
// /api/browse/category-overlap unchanged, only the rendering differs.
// Copy-adapted from RelatednessGraph.jsx, not imported from it: that
// component is ego-anchored (one center + its top-K neighbors, reloaded
// per K); this is a small, complete graph of ALL of one level's siblings
// at once, no center node, no top-K control -- different enough that
// forcing a shared abstraction would cost more than it saves.
function CategoryOverlapGraph({ bucket, parent, basePath }) {
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
      // Without this, nodes settle relative to whatever (possibly tiny,
      // pre-layout) size the canvas had on its FIRST render, then never
      // re-center once the container reports its real size -- same fix
      // RelatednessGraph.jsx's own ResizeObserver applies.
      setTimeout(() => fgRef.current?.zoomToFit(ZOOM_MS, ZOOM_PADDING), ZOOM_MS)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    setData(null)
    setError('')
    const params = new URLSearchParams()
    if (bucket) params.set('bucket', bucket)
    if (parent) params.set('parent', parent)
    fetch(`${API_BASE}/api/browse/category-overlap?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then(setData)
      .catch((err) => setError(err.message || 'failed to load overlap'))
  }, [bucket, parent])

  // The 6-bucket top level's sibling "codes" are bucket keys themselves --
  // /api/browse/words takes those as `domain`. Every other level's
  // siblings are real USAS codes, filtered via `all_top_code` (an AND:
  // "carries a category under BOTH" -- deliberately not `top_code`, which
  // is an OR and would show either category's words, not the overlap).
  const isTopLevel = !bucket && !parent

  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] }
    return {
      nodes: data.sizes.map((s) => ({ id: s.code, name: s.name, word_count: s.word_count })),
      links: data.cells.map((c) => ({
        source: c.code_a, target: c.code_b, ratio: c.ratio, shared_words: c.shared_words,
      })),
    }
  }, [data])

  const maxCount = Math.max(1, ...graphData.nodes.map((n) => n.word_count))
  const maxRatio = Math.max(0, ...graphData.links.map((l) => l.ratio))

  // Disconnected nodes (no edge -- the backend only returns cells that
  // actually overlap) rely on charge repulsion alone, boosted well past
  // react-force-graph's default (way too weak for 6-28px circles) so they
  // don't start stacked on each other either.
  //
  // A single zoomToFit right after reheating races the simulation (same
  // problem RelatednessGraph.jsx's own loadGraph hit): the boosted charge/
  // link-distance forces above keep pushing nodes apart for a couple of
  // seconds, so one immediate fit locks the viewport to an early, still-
  // mid-expansion snapshot -- nodes then drift outside it as the layout
  // keeps opening up (confirmed live: the graph was still stuck in one
  // corner, half off-canvas, seconds after mount). Chase it instead:
  // re-fit every 300ms for a few seconds so the view tracks the layout as
  // it actually expands; onEngineStop's own zoomToFit is the final
  // correction once the simulation naturally cools down.
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

  function overlapHref(codeA, codeB) {
    const params = new URLSearchParams()
    // Both AND-intersections, matching exactly what this graph's own edge
    // (browse_category_overlap's shared-word count) represents -- `domain`
    // alone is an OR (right for the facet-row bucket chips, wrong here: a
    // real live check found it showing the ~40k-word union of two buckets'
    // entire membership for an edge whose own count was in the hundreds).
    const key = isTopLevel ? 'all_domain' : 'all_top_code'
    params.append(key, codeA)
    params.append(key, codeB)
    return `/app/words?${params}`
  }

  // Node fill: real bucket colors at the top level (the one place 6 truly
  // distinct hues carry information); a flat single color everywhere
  // deeper, where every sibling already shares the same bucket, so
  // per-node bucket coloring would just repaint every node identically.
  const nodeFill = useCallback(
    (node) => (isTopLevel ? colorForBucket(node.id) : bucket ? colorForBucket(bucket) : cssVar('--accent', '#1c6dbd')),
    [isTopLevel, bucket],
  )

  const nodeRadius = useCallback((node) => radiusForCount(node.word_count, maxCount), [maxCount])

  // Shared by paintNode (visible canvas) and paintNodePointerArea (force-
  // graph's separate hit-test canvas) -- without the latter, clicks miss
  // everywhere but the exact center pixel (force-graph's default hit
  // radius comes from nodeRelSize, not whatever paintNode actually drew;
  // confirmed live in RelatednessGraph.jsx's own build of this component).
  const paintNode = useCallback(
    (node, ctx, globalScale) => {
      const r = nodeRadius(node)
      ctx.beginPath()
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
      ctx.fillStyle = nodeFill(node)
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
    [nodeRadius, nodeFill],
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

  function handleNodeClick(node) {
    navigate(`${basePath}/${encodeURIComponent(node.id)}`)
  }

  function handleLinkClick(link) {
    navigate(overlapHref(endpointId(link.source), endpointId(link.target)))
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
      {/* Always mounted, even during loading/error/empty states -- the
          ResizeObserver effect above runs once on mount with `[]` deps and
          bails out permanently if containerRef.current is null at that
          instant, which it would be if this div were behind an early
          return gated on `data`. Confirmed live: with an early return here,
          dims never left {0,0} and ForceGraph2D silently fell back to its
          own default size (Playwright's 1280x720 viewport, cropped by the
          wrap's own much smaller overflow:hidden box) instead of the
          container's real ~886x420. */}
      <div className="category-overlap-canvas-wrap" ref={containerRef}>
        {error && <div className="error-banner">{error}</div>}
        {!error && !data && <div className="page-loading">Loading…</div>}
        {!error && showEmpty && (
          <p className="category-empty">Not enough sibling categories here to compare overlap.</p>
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
            // A graph this small (<=15 nodes) doesn't need d3-force's default
            // 15s/unlimited-tick cooldown budget to converge -- faster decay +
            // a bounded tick count settles it in well under a second instead
            // of leaving it visibly drifting (and un-fitted) for seconds
            // after the boosted charge/link-distance force change above.
            d3AlphaDecay={0.05}
            cooldownTicks={150}
          />
        )}
      </div>
      {!error && data && !showEmpty && (
        <p className="category-empty">
          Circle size = word count · line thickness = overlap strength. Click a circle to open that category, a line
          to browse its shared words.
        </p>
      )}
    </div>
  )
}

export default CategoryOverlapGraph
