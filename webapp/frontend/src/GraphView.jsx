import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import GraphLegend from './GraphLegend'
import { colorForBucket } from './domainColors'
import { ZOOM_MS, ZOOM_PADDING, cssVar, radiusForZipf } from './graphUtils'
import './GraphView.css'

// Lazy: pulls in three.js (WebGL, large) only when the user actually opens
// 3D mode — see GraphView3D.jsx's own note on why this split exists.
const GraphView3D = lazy(() => import('./GraphView3D'))

const API_BASE = ''

const ROTATE_STEP = Math.PI / 8 // 22.5° per click
const ROTATE_PHI_MIN = 0.1 // radians off the top pole
const ROTATE_PHI_MAX = Math.PI - 0.1 // radians off the bottom pole

function GraphView({ initialWordId, onNodeNavigate, hideSearch = false }) {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [center, setCenter] = useState(null) // { id, lemma }
  const [signal, setSignal] = useState('definition')
  const [mode, setMode] = useState('2d') // '2d' | '3d'
  const [rawGraph, setRawGraph] = useState({ nodes: [], edges: [] })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dims, setDims] = useState({ width: 0, height: 0 })
  const [isFullscreen, setIsFullscreen] = useState(false)
  const searchRef = useRef(null)
  const viewRef = useRef(null) // fullscreen target: whole view, so search/controls stay usable
  const containerRef = useRef(null) // canvas wrap: measured for the 2D/3D renderer's explicit width/height
  const fgRef = useRef(null)

  // close the suggestions dropdown on outside click or Escape. mousedown (not
  // blur) so clicking a suggestion doesn't close the list before onClick fires.
  useEffect(() => {
    function handlePointerDown(e) {
      if (searchRef.current && !searchRef.current.contains(e.target)) setSuggestions([])
    }
    function handleKeyDown(e) {
      if (e.key === 'Escape') setSuggestions([])
    }
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  // ForceGraph2D defaults its canvas to window.innerWidth/innerHeight, not its
  // parent's size — without this it renders far larger than the (clipped)
  // container, so zoomToFit "centers" relative to space the user can't see.
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

  // debounced search-as-you-type against /api/words/search
  useEffect(() => {
    if (!query.trim()) {
      setSuggestions([])
      return
    }
    const handle = setTimeout(() => {
      fetch(`${API_BASE}/api/words/search?q=${encodeURIComponent(query)}&limit=10`)
        .then((res) => res.json())
        .then(setSuggestions)
        .catch(() => {})
    }, 200)
    return () => clearTimeout(handle)
  }, [query])

  const loadGraph = useCallback((wordId, sig) => {
    setLoading(true)
    setError('')
    fetch(`${API_BASE}/api/words/${wordId}/graph?signal=${sig}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => {
        setCenter(data.center)
        setRawGraph({ nodes: data.nodes, edges: data.edges })
        // onEngineStop's zoomToFit only fires once the sim fully cools down
        // (default 15s) — that's a long time to show an off-center graph, so
        // fit early too (the layout is mostly settled well before cooldown).
        setTimeout(() => fgRef.current?.zoomToFit(ZOOM_MS, ZOOM_PADDING), ZOOM_MS)
      })
      .catch((err) => setError(err.message || 'failed to load graph'))
      .finally(() => setLoading(false))
  }, [])

  // Embedded usage (WordDetail) pre-loads a starting word instead of making
  // the user search for it. Keyed on the prop, not mount-only: clicking a
  // node inside an embedded graph navigates WordDetail to a new /words/:id,
  // and React Router keeps this component mounted across that navigation —
  // only initialWordId changes, so the effect must re-fire to follow it.
  useEffect(() => {
    if (initialWordId != null) loadGraph(initialWordId, signal)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialWordId])

  function handlePick(word) {
    setQuery(word.lemma)
    setSuggestions([])
    loadGraph(word.id, signal)
  }

  function handleSignalChange(next) {
    setSignal(next)
    if (center) loadGraph(center.id, next)
  }

  function handleNodeClick(node) {
    if (node.id === center?.id) return
    if (onNodeNavigate) {
      onNodeNavigate(node)
      return
    }
    setQuery(node.lemma)
    loadGraph(node.id, signal)
  }

  // Rebuilt with fresh node/link objects on every mode switch, not just on
  // data reload. 2D's simulation mutates node.x/y in place and resolves each
  // link's source/target from a string id into a direct object reference —
  // reusing those same objects for 3D would start its simulation from an
  // already-settled, nearly-flat 2D layout (and hand it links pointing at
  // stale 2D node objects) instead of spreading nodes freely in z.
  const graphData = useMemo(
    () => ({
      nodes: rawGraph.nodes.map((n) => ({ ...n })),
      links: rawGraph.edges.map((e) => ({ source: e.source, target: e.target, distance: e.distance })),
    }),
    [rawGraph, mode],
  )

  // Explicit rotate-along-x/y controls for 3D: read the camera's current
  // spherical position off its live THREE.Camera, nudge azimuth (theta, the
  // "y-axis"/left-right swing) or polar angle (phi, the "x-axis"/up-down
  // tilt), then hand the new position to cameraPosition(). Polar is clamped
  // away from the poles so successive up/down clicks can't flip the view
  // through straight-overhead and reverse the drag direction the user expects.
  const rotateCamera = useCallback((dTheta, dPhi) => {
    const graph = fgRef.current
    if (!graph?.camera) return
    const { x, y, z } = graph.camera().position
    const r = Math.sqrt(x * x + y * y + z * z) || 1
    const theta = Math.atan2(x, z) + dTheta
    const phi = Math.min(
      ROTATE_PHI_MAX,
      Math.max(ROTATE_PHI_MIN, Math.acos(Math.max(-1, Math.min(1, y / r))) + dPhi),
    )
    graph.cameraPosition(
      { x: r * Math.sin(phi) * Math.sin(theta), y: r * Math.cos(phi), z: r * Math.sin(phi) * Math.cos(theta) },
      { x: 0, y: 0, z: 0 },
      300,
    )
  }, [])

  const paintNode = useMemo(
    () => (node, ctx, globalScale) => {
      const r = node.id === center?.id ? Math.max(radiusForZipf(node.zipf), 10) : radiusForZipf(node.zipf)
      ctx.beginPath()
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
      ctx.fillStyle = colorForBucket(node.color_bucket)
      ctx.fill()
      const textColor = cssVar('--text-h', '#08060d')
      if (node.id === center?.id) {
        ctx.lineWidth = 2 / globalScale
        ctx.strokeStyle = textColor
        ctx.stroke()
      }
      const fontSize = Math.max(10 / globalScale, 2)
      ctx.font = `${fontSize}px sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillStyle = textColor
      ctx.fillText(node.lemma, node.x, node.y + r + 1)
    },
    [center],
  )

  // Shared between 2D and 3D — same hover tooltip either way.
  const nodeLabel = useCallback(
    (n) => `<b>${n.lemma}</b><br/>${n.definition ?? ''}<br/>${n.usas_name ?? 'Uncategorized'} · zipf ${n.zipf.toFixed(1)}`,
    [],
  )
  const handleEngineStop = useCallback(() => fgRef.current?.zoomToFit(ZOOM_MS, ZOOM_PADDING), [])

  return (
    <div className="graph-view" ref={viewRef}>
      <div className="graph-controls">
        {!hideSearch && (
          <div className="graph-search" ref={searchRef}>
            <input
              type="text"
              placeholder="Search for a word…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            {suggestions.length > 0 && (
              <ul className="graph-suggestions">
                {suggestions.map((w) => (
                  <li key={w.id} onClick={() => handlePick(w)}>
                    {w.lemma}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
        <div className="graph-signal-toggle">
          <button className={signal === 'definition' ? 'active' : ''} onClick={() => handleSignalChange('definition')}>
            Meaning
          </button>
          <button className={signal === 'fasttext' ? 'active' : ''} onClick={() => handleSignalChange('fasttext')}>
            Spelling
          </button>
        </div>
        <div className="graph-signal-toggle">
          <button className={mode === '2d' ? 'active' : ''} onClick={() => setMode('2d')}>
            2D
          </button>
          <button className={mode === '3d' ? 'active' : ''} onClick={() => setMode('3d')}>
            3D
          </button>
        </div>
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>

      <GraphLegend />

      {error && <div className="graph-error">{error}</div>}
      {loading && <div className="graph-loading">Loading…</div>}

      <div className="graph-canvas-wrap" ref={containerRef}>
        {graphData.nodes.length > 0 && mode === '2d' && (
          <ForceGraph2D
            ref={fgRef}
            width={dims.width || undefined}
            height={dims.height || undefined}
            graphData={graphData}
            nodeLabel={nodeLabel}
            nodeCanvasObject={paintNode}
            nodeCanvasObjectMode={() => 'replace'}
            linkColor={() => cssVar('--border', '#e5e4e7')}
            onNodeClick={handleNodeClick}
            onEngineStop={handleEngineStop}
          />
        )}
        {graphData.nodes.length > 0 && mode === '3d' && (
          <Suspense fallback={<div className="graph-loading">Loading 3D…</div>}>
            <GraphView3D
              ref={fgRef}
              width={dims.width || undefined}
              height={dims.height || undefined}
              graphData={graphData}
              center={center}
              nodeLabel={nodeLabel}
              onNodeClick={handleNodeClick}
              onEngineStop={handleEngineStop}
            />
          </Suspense>
        )}
        {graphData.nodes.length === 0 && !loading && (
          <div className="graph-empty">Search for a word to see its similarity graph.</div>
        )}
        {graphData.nodes.length > 0 && mode === '3d' && (
          <div className="graph-rotate-controls" role="group" aria-label="Rotate 3D graph">
            <button type="button" aria-label="Rotate up" onClick={() => rotateCamera(0, -ROTATE_STEP)}>
              ▲
            </button>
            <div className="graph-rotate-row">
              <button type="button" aria-label="Rotate left" onClick={() => rotateCamera(-ROTATE_STEP, 0)}>
                ◀
              </button>
              <button type="button" aria-label="Rotate right" onClick={() => rotateCamera(ROTATE_STEP, 0)}>
                ▶
              </button>
            </div>
            <button type="button" aria-label="Rotate down" onClick={() => rotateCamera(0, ROTATE_STEP)}>
              ▼
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default GraphView
