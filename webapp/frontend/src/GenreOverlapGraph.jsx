import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { useNavigate } from 'react-router-dom'
import { cssVar, ZOOM_MS, ZOOM_PADDING } from './graphUtils'
import { colorForCluster } from './clusterColors'
import './GraphView.css'
import './Categories.css'

const API_BASE = ''
const NODE_RADIUS_MIN = 6
const NODE_RADIUS_MAX = 28
const LINK_WIDTH_MIN = 1
const LINK_WIDTH_MAX = 6
const LINK_DISTANCE = 2 * NODE_RADIUS_MAX + 60
const CHARGE_BASE = -220 // this graph's own baseline (see chase-zoom effect below), not RelatednessGraph's CHARGE_BASE -- the two graphs were tuned independently against different node/edge densities
const SPREAD_MIN = 0.3
const SPREAD_MAX = 5
const SPREAD_DEFAULT = 1
const ZOOM_MIN = 0.1
const ZOOM_MAX = 8
const ZOOM_DEFAULT = 1
const MIN_SHARED_WORDS_DEFAULT = 0
const INDIRECT_LEVEL_OPTIONS = [0, 1, 2]

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
//
// The fullscreen/Zoom/Spread machinery below is copy-adapted from
// RelatednessGraph.jsx (see its own comments for the full reasoning on the
// programmatic-vs-user-zoom guard) -- this graph has no ego-fetch-per-node
// step, but once a genre's LABEL is clicked (as opposed to its circle,
// which still just navigates to that genre's books) it filters down to an
// ego-subgraph of that node the same way RelatednessGraph's loadGraph
// swaps in a new node set, so it needs the identical "was this a genuine
// new node set or just a Spread tweak on the same one" distinction.
function GenreOverlapGraph() {
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [dims, setDims] = useState({ width: 0, height: 0 })
  const [spread, setSpread] = useState(SPREAD_DEFAULT)
  const [zoomLevel, setZoomLevel] = useState(ZOOM_DEFAULT)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [focusId, setFocusId] = useState(null)
  const [indirectLevels, setIndirectLevels] = useState(0)
  const [minSharedWords, setMinSharedWords] = useState(MIN_SHARED_WORDS_DEFAULT)
  const viewRef = useRef(null)
  const containerRef = useRef(null)
  const fgRef = useRef(null)
  const chaseIntervalRef = useRef(null)
  // See RelatednessGraph.jsx's identical refs for the full reasoning --
  // once the user has manually zoomed/panned/dragged the Zoom slider,
  // automatic re-fits (chase loop, onEngineStop, resize) back off until a
  // genuinely new node set (a fresh focus, or clearing one) loads.
  const userAdjustedRef = useRef(false)
  const programmaticZoomRef = useRef(false)
  const programmaticZoomTimeoutRef = useRef(null)

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

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/genre-overlap`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then(setData)
      .catch((err) => setError(err.message || 'failed to load overlap'))
  }, [])

  // The full, unfiltered graph -- ego-focusing (below) narrows this down
  // for DISPLAY only; radius/color scales stay pinned to the full set so a
  // node's size and color never jump when focusing/unfocusing.
  const fullGraphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] }
    return {
      nodes: data.sizes.map((s) => ({ id: s.genre, name: s.genre, word_count: s.word_count })),
      links: data.cells.map((c) => ({
        source: c.genre_a, target: c.genre_b, ratio: c.ratio, shared_words: c.shared_words,
      })),
    }
  }, [data])

  const maxCount = Math.max(1, ...fullGraphData.nodes.map((n) => n.word_count))
  // Ceiling for the Min shared words control -- off the UNFILTERED set, so
  // raising the cutoff never shrinks the range out from under the input
  // the user's already sitting on.
  const maxSharedWords = Math.max(0, ...fullGraphData.links.map((l) => l.shared_words))

  // Density control: relationships below the cutoff are dropped as EDGES
  // only -- every genre stays visible (word count/color don't depend on its
  // edges), only which overlap lines get drawn does. All nodes fitting in
  // one graph already (genre has no "top N", see the module docstring)
  // means edge count is the only thing that can make this unreadable as
  // the corpus grows, so this is the knob for that, not a node filter.
  const edgeFilteredGraphData = useMemo(
    () => ({
      nodes: fullGraphData.nodes,
      links: fullGraphData.links.filter((l) => l.shared_words >= minSharedWords),
    }),
    [fullGraphData, minSharedWords],
  )
  // Scoped to the cutoff but NOT to the current focus, so a link's width
  // stays stable as you focus/unfocus at the same cutoff -- only changing
  // the cutoff itself should redistribute the width scale.
  const maxRatio = Math.max(0, ...edgeFilteredGraphData.links.map((l) => l.ratio))

  // Stable index into data.sizes (already alphabetically sorted by the
  // backend) so a genre keeps the same color across reloads/focus changes,
  // rather than a color depending on this render's node ordering.
  const colorForGenre = useMemo(() => {
    const map = new Map(fullGraphData.nodes.map((n, i) => [n.id, colorForCluster(i)]))
    return (id) => map.get(id) || cssVar('--accent', '#1c6dbd')
  }, [fullGraphData])

  // Clicking a LABEL (see handleNodeClick) ego-focuses the graph on that
  // genre: only nodes reachable within `indirectLevels + 1` hops (the
  // clicked genre counts as hop 0, its direct overlap partners as hop 1,
  // "indirect" partners beyond that) stay visible, same shape as
  // RelatednessGraph's server-side ego graph but computed client-side
  // since /api/browse/genre-overlap already hands back the complete
  // (small, ~40-node) graph in one request -- no need for a second
  // endpoint just to pre-filter it. Hops are traced over the CUTOFF-filtered
  // edges (edgeFilteredGraphData), not the raw full set -- at the current
  // corpus scale every genre shares at least one word with every other, so
  // without the cutoff already thinning the graph, "+0/+1/+2 indirect"
  // wouldn't narrow anything (every genre is already a 1-hop neighbor of
  // every other). Min shared words is what makes hop-depth meaningful.
  const visibleGraphData = useMemo(() => {
    if (!focusId) return edgeFilteredGraphData
    const adjacency = new Map()
    edgeFilteredGraphData.links.forEach((l) => {
      const a = endpointId(l.source)
      const b = endpointId(l.target)
      if (!adjacency.has(a)) adjacency.set(a, [])
      if (!adjacency.has(b)) adjacency.set(b, [])
      adjacency.get(a).push(b)
      adjacency.get(b).push(a)
    })
    const maxHops = 1 + indirectLevels
    const hops = new Map([[focusId, 0]])
    const queue = [focusId]
    while (queue.length) {
      const cur = queue.shift()
      const d = hops.get(cur)
      if (d >= maxHops) continue
      for (const nb of adjacency.get(cur) || []) {
        if (!hops.has(nb)) {
          hops.set(nb, d + 1)
          queue.push(nb)
        }
      }
    }
    return {
      nodes: edgeFilteredGraphData.nodes.filter((n) => hops.has(n.id)),
      links: edgeFilteredGraphData.links.filter(
        (l) => hops.has(endpointId(l.source)) && hops.has(endpointId(l.target)),
      ),
    }
  }, [edgeFilteredGraphData, focusId, indirectLevels])

  // Shared by a fresh data/focus load AND the Spread slider itself, same
  // split RelatednessGraph.jsx uses -- see its own comment for why: dragging
  // Spread should visibly loosen/tighten the SAME node set immediately, not
  // trigger a refetch or a fresh auto-fit fight with a camera the user has
  // already taken control of.
  const applyForcesAndChaseZoom = useCallback(
    (nodeCount) => {
      fgRef.current?.d3Force('link')?.distance(LINK_DISTANCE * spread)
      fgRef.current?.d3Force('charge')?.strength(CHARGE_BASE * spread)
      fgRef.current?.d3ReheatSimulation?.()
      if (chaseIntervalRef.current) clearInterval(chaseIntervalRef.current)
      if (userAdjustedRef.current) return
      let elapsed = 0
      const chaseBudgetMs = Math.min(3000 + nodeCount * 40, 8000)
      chaseIntervalRef.current = setInterval(() => {
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

  // A genuinely new node set (initial load, a fresh focus, clearing one, or
  // changing how many indirect levels are shown) deserves a fresh auto-fit
  // even if the user had taken control of the PREVIOUS camera -- only a
  // Spread tweak on the SAME set (the effect below) respects that.
  useEffect(() => {
    if (visibleGraphData.nodes.length === 0) return
    userAdjustedRef.current = false
    const timer = setTimeout(() => applyForcesAndChaseZoom(visibleGraphData.nodes.length), 50)
    return () => {
      clearTimeout(timer)
      if (chaseIntervalRef.current) clearInterval(chaseIntervalRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibleGraphData])

  useEffect(() => {
    if (visibleGraphData.nodes.length > 0) applyForcesAndChaseZoom(visibleGraphData.nodes.length)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spread])

  const nodeRadius = useCallback((node) => radiusForCount(node.word_count, maxCount), [maxCount])

  const paintNode = useCallback(
    (node, ctx, globalScale) => {
      const r = nodeRadius(node)
      const isFocused = node.id === focusId
      ctx.beginPath()
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
      ctx.fillStyle = colorForGenre(node.id)
      ctx.fill()
      ctx.lineWidth = (isFocused ? 2.5 : 1.5) / globalScale
      ctx.strokeStyle = isFocused ? cssVar('--text-h', '#08060d') : cssVar('--bg', '#fff')
      ctx.stroke()
      const fontSize = Math.max(11 / globalScale, 3)
      ctx.font = `${fontSize}px sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillStyle = cssVar('--text-h', '#08060d')
      ctx.fillText(node.name, node.x, node.y + r + 3)
    },
    [nodeRadius, colorForGenre, focusId],
  )

  // Paints BOTH the circle AND the label's bounding box into the same
  // node's hit-test color, so a click landing on the label text -- well
  // outside the circle's own radius -- still resolves to onNodeClick for
  // this node instead of missing entirely (force-graph only makes a region
  // clickable if something was painted there on its hidden pointer-area
  // canvas). handleNodeClick then tells circle vs. label apart itself,
  // from where the click actually landed relative to the node's center.
  const paintNodePointerArea = useCallback(
    (node, color, ctx, globalScale) => {
      const r = nodeRadius(node)
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
      ctx.fill()
      const fontSize = Math.max(11 / globalScale, 3)
      ctx.font = `${fontSize}px sans-serif`
      const textWidth = ctx.measureText(node.name).width
      const pad = 2 / globalScale
      ctx.fillRect(node.x - textWidth / 2 - pad, node.y + r, textWidth + pad * 2, fontSize + pad * 2)
    },
    [nodeRadius],
  )

  // A genre node's CIRCLE -> the Books page filtered to it (unchanged
  // behavior); its LABEL -> ego-focuses the graph on that node instead,
  // per Brian's ask. Distinguishes the two from the click's own graph-space
  // y-coordinate rather than from a second onClick prop -- force-graph only
  // exposes one click callback per node, see paintNodePointerArea above for
  // why both regions register a hit at all. `node.y + r` is exactly the
  // pointer area's own circle/label boundary, so this needs no radius
  // fudge-factor: every point inside the circle satisfies y <= node.y + r
  // by definition, so a hit with y > node.y + r can only have landed in the
  // label rect. A radius-based distance check instead (an earlier version
  // of this used `dist <= r*1.15`) breaks at high zoom: the label rect's
  // height shrinks in graph-space as globalScale grows (paintNodePointerArea
  // keeps its ON-SCREEN size constant the same way paintNode's font does),
  // while a fixed multiple of r does not, so a large node's whole visible
  // label eventually falls inside the radius test and clicking it navigates
  // instead of focusing.
  function handleNodeClick(node, event) {
    const r = nodeRadius(node)
    const coords = event ? fgRef.current?.screen2GraphCoords(event.offsetX, event.offsetY) : null
    const isLabelClick = coords && coords.y > node.y + r
    if (isLabelClick) {
      setFocusId(node.id)
    } else {
      navigate(`/app/books?genre=${encodeURIComponent(node.id)}`)
    }
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
  const handleEngineStop = useCallback(() => {
    if (userAdjustedRef.current) return
    doZoomToFit(ZOOM_MS, ZOOM_PADDING)
  }, [doZoomToFit])

  // See RelatednessGraph.jsx's identical handler for the full reasoning:
  // fires for every zoom-transform change, automatic or user-driven, and
  // is the one place that tells the two apart (programmaticZoomRef is only
  // set around our OWN doZoomToFit calls) so future auto-fits know whether
  // to back off.
  const handleZoom = useCallback((transform) => {
    setZoomLevel(Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, transform.k)))
    if (!programmaticZoomRef.current) {
      userAdjustedRef.current = true
      if (chaseIntervalRef.current) clearInterval(chaseIntervalRef.current)
    }
  }, [])

  const showEmpty = data && data.sizes.length < 2
  const focusedNode = focusId ? fullGraphData.nodes.find((n) => n.id === focusId) : null

  return (
    <div className="graph-view" ref={viewRef}>
      <div className="graph-controls">
        {focusedNode ? (
          <div className="graph-signal-toggle" role="group" aria-label="Indirect levels shown around the focused genre">
            <span className="graph-range-control">Focused on {focusedNode.name}</span>
            {INDIRECT_LEVEL_OPTIONS.map((n) => (
              <button
                key={n}
                type="button"
                className={indirectLevels === n ? 'active' : ''}
                onClick={() => setIndirectLevels(n)}
              >
                +{n} indirect
              </button>
            ))}
            <button type="button" onClick={() => setFocusId(null)}>
              Show all genres
            </button>
          </div>
        ) : (
          <span className="graph-range-control">Click a genre's label to focus on it</span>
        )}
        <label className="graph-range-control">
          Min shared words
          <input
            type="number"
            min={0}
            max={maxSharedWords}
            step={1}
            value={minSharedWords}
            onChange={(e) => setMinSharedWords(Math.min(maxSharedWords, Math.max(0, Number(e.target.value) || 0)))}
            aria-label="Minimum shared words -- hide overlap lines below this count to manage density"
          />
        </label>
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
      {!error && !data && <div className="graph-loading">Loading…</div>}
      {!error && showEmpty && <p className="graph-empty">Not enough genres tagged yet to compare overlap.</p>}

      <div className="graph-canvas-wrap" ref={containerRef}>
        {!error && data && !showEmpty && (
          <ForceGraph2D
            ref={fgRef}
            width={dims.width || undefined}
            height={dims.height || undefined}
            graphData={visibleGraphData}
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
            onZoom={handleZoom}
            d3AlphaDecay={0.05}
            cooldownTicks={150}
          />
        )}
      </div>
      {!error && data && !showEmpty && (
        <p className="category-empty">
          Circle size = word count · line thickness = overlap strength. Click a circle to browse that genre's
          books, its label to focus the graph on just that genre.
          {minSharedWords > 0 &&
            ` Hiding ${(fullGraphData.links.length - edgeFilteredGraphData.links.length).toLocaleString()} of ${fullGraphData.links.length.toLocaleString()} relationships below ${minSharedWords.toLocaleString()} shared words.`}
        </p>
      )}
    </div>
  )
}

export default GenreOverlapGraph
