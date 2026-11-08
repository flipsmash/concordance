import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { colorForBucket } from './domainColors'
import { cssVar } from './graphUtils'
import './DomainMap.css'
import './GraphView.css' // .graph-controls/.graph-maximize/.graph-signal-toggle/.graph-error/.graph-loading/.graph-empty

const API_BASE = ''
const PADDING = 28
const RADIUS_MIN = 1.6
const RADIUS_MAX = 6
const HIT_PX = 8

const ENTITY_TABS = [
  { id: 'book', label: 'Works' },
  { id: 'author', label: 'Authors' },
]

const SPREAD_MIN = 0.5
const SPREAD_MAX = 4
const SPREAD_DEFAULT = 1

// Relationship map by DISCIPLINE-CATEGORY DISTRIBUTION, not shared vocabulary
// -- distinct from every other map/graph/matrix in this app (all built on
// author_similarity/book_similarity's word overlap). Two works can sit close
// here with no words in common, as long as their vocabulary leans on the
// same mix of USAS discourse fields (e.g. both heavy in Nature & Science).
// See /api/browse/domain-map for the PCA + corpus-relative-lift computation.
//
// Canvas, not SVG: ~29k points (SVG re-rendered all of them on every hover).
// Colour by EMPHASIS, not six hues at once: in a scatter any two dots can sit
// side by side, and no six categorical colours stay distinguishable across
// every pair (the dataviz validator fails all-pairs past three -- the old
// palette had orange vs red at normal-vision dE 7.1). So every dot is a
// neutral grey showing the overall shape, and picking a field draws that
// field's dots on top in its own hue.
function DomainMap() {
  const navigate = useNavigate()
  const [entity, setEntity] = useState('book')
  const [nodes, setNodes] = useState(null) // null = loading
  const [error, setError] = useState('')
  const [hovered, setHovered] = useState(null) // {p, x, y}
  const [legend, setLegend] = useState([])
  const [focus, setFocus] = useState(null) // a domain bucket, or null = none highlighted
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [spread, setSpread] = useState(SPREAD_DEFAULT)
  const [size, setSize] = useState({ w: 0, h: 0 })
  const [themeTick, setThemeTick] = useState(0)
  const containerRef = useRef(null)
  const wrapRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/graph/legend`).then((res) => res.json()).then(setLegend).catch(() => {})
  }, [])

  useEffect(() => {
    setNodes(null)
    setError('')
    setHovered(null)
    fetch(`${API_BASE}/api/browse/domain-map?entity=${entity}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => setNodes(data.nodes))
      .catch((err) => setError(err.message || 'failed to load domain map'))
  }, [entity])

  useEffect(() => {
    function handleFullscreenChange() {
      setIsFullscreen(document.fullscreenElement === containerRef.current)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => setThemeTick((t) => t + 1)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return undefined
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [nodes])

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen()
    else containerRef.current?.requestFullscreen()
  }

  // Layout: fit the full extent to the canvas, then `spread` magnifies
  // uniformly about the centre (a true zoom -- every pairwise distance grows
  // by the same factor, so a crowded middle becomes individually visible).
  const points = useMemo(() => {
    if (!nodes || !nodes.length || !size.w) return []
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, maxWords = 1
    for (const n of nodes) {
      minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x)
      minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y)
      maxWords = Math.max(maxWords, n.word_count)
    }
    const span = Math.max(maxX - minX, maxY - minY) || 1
    const scale = (Math.min(size.w, size.h) - 2 * PADDING) / span * spread
    const cx0 = (minX + maxX) / 2, cy0 = (minY + maxY) / 2
    return nodes.map((n) => ({
      ...n,
      cx: size.w / 2 + (n.x - cx0) * scale,
      cy: size.h / 2 - (n.y - cy0) * scale,
      r: RADIUS_MIN + Math.sqrt(n.word_count / maxWords) * (RADIUS_MAX - RADIUS_MIN), // area-proportional
    }))
  }, [nodes, size, spread])

  // Draw: everything in grey, then the focused field's dots on top in colour.
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !size.w) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(size.w * dpr)
    canvas.height = Math.round(size.h * dpr)
    canvas.style.width = `${size.w}px`
    canvas.style.height = `${size.h}px`
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size.w, size.h)
    const grey = cssVar('--text', '#5b6572')
    const dot = (p) => {
      ctx.beginPath()
      ctx.arc(p.cx, p.cy, p.r, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.fillStyle = grey
    ctx.globalAlpha = focus ? 0.12 : 0.28
    for (const p of points) if (p.dominant_bucket !== focus) dot(p)
    if (focus) {
      ctx.globalAlpha = 0.85
      ctx.fillStyle = colorForBucket(focus)
      for (const p of points) if (p.dominant_bucket === focus) dot(p)
    }
    ctx.globalAlpha = 1
    if (hovered) {
      ctx.lineWidth = 2
      ctx.strokeStyle = cssVar('--text-h', '#10181f')
      ctx.beginPath()
      ctx.arc(hovered.p.cx, hovered.p.cy, hovered.p.r + 2, 0, Math.PI * 2)
      ctx.stroke()
    }
    // themeTick: redraw with the other mode's colours
  }, [points, size, focus, hovered, themeTick]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const id = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(id)
  }, [draw])

  // Hover: nearest dot within reach, via a coarse grid over screen space.
  const grid = useMemo(() => {
    const cell = 16
    const cells = new Map()
    points.forEach((p, i) => {
      const k = `${Math.floor(p.cx / cell)},${Math.floor(p.cy / cell)}`
      if (!cells.has(k)) cells.set(k, [])
      cells.get(k).push(i)
    })
    return { cell, cells }
  }, [points])

  function hit(x, y) {
    const { cell, cells } = grid
    let best = null, bestD = HIT_PX * HIT_PX
    for (let cx = Math.floor((x - HIT_PX) / cell); cx <= Math.floor((x + HIT_PX) / cell); cx++)
      for (let cy = Math.floor((y - HIT_PX) / cell); cy <= Math.floor((y + HIT_PX) / cell); cy++)
        for (const i of cells.get(`${cx},${cy}`) || []) {
          const p = points[i]
          if (focus && p.dominant_bucket !== focus) continue
          const d = (p.cx - x) ** 2 + (p.cy - y) ** 2
          if (d <= bestD) { bestD = d; best = p }
        }
    return best
  }

  function onMove(e) {
    const r = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - r.left, y = e.clientY - r.top
    const p = hit(x, y)
    setHovered(p ? { p, x, y } : null)
  }

  function onClick() {
    const p = hovered?.p
    if (!p) return
    if (entity === 'book') navigate(`/app/authors/${encodeURIComponent(p.subtitle || '')}/${p.id}`)
    else navigate(`/app/authors/${encodeURIComponent(p.id)}`)
  }

  const counts = useMemo(() => {
    const c = {}
    for (const n of nodes || []) c[n.dominant_bucket] = (c[n.dominant_bucket] || 0) + 1
    return c
  }, [nodes])
  const noun = entity === 'book' ? 'works' : 'authors'

  return (
    <div className="domain-map-page">
      <header className="domain-map-header">
        <Link to="/app/visualizations" className="domain-map-crumb">← Visualizations</Link>
        <h1>Works &amp; authors by discipline category</h1>
        <p className="muted">
          Each dot is a work (or author), placed by how its vocabulary spreads across subject fields
          &mdash; science, law, the arts&hellip; Dots close together lean on the same mix of fields, even with
          no words in common. Pick a field to see where its {noun} sit.
        </p>
      </header>

      <div className="domain-map" ref={containerRef}>
        <div className="graph-controls domain-map-controls">
          <div className="graph-signal-toggle" role="group" aria-label="Entity">
            {ENTITY_TABS.map((t) => (
              <button key={t.id} type="button" className={entity === t.id ? 'active' : ''} onClick={() => setEntity(t.id)}>
                {t.label}
              </button>
            ))}
          </div>
          <label className="domain-map-spread">
            Spread
            <input type="range" min={SPREAD_MIN} max={SPREAD_MAX} step={0.1} value={spread}
              onChange={(e) => setSpread(Number(e.target.value))} />
          </label>
          <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
            {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
          </button>
        </div>

        <div className="domain-map-fields" role="group" aria-label="Highlight a field">
          <span className="muted">Highlight:</span>
          {legend.map((e) => (
            <button key={e.bucket} type="button" aria-pressed={focus === e.bucket}
              className={`domain-map-field${focus === e.bucket ? ' active' : ''}`}
              onClick={() => setFocus(focus === e.bucket ? null : e.bucket)}>
              <span className="domain-map-legend-swatch" style={{ background: colorForBucket(e.bucket) }} />
              {e.name}
              {nodes && <span className="muted"> {(counts[e.bucket] || 0).toLocaleString()}</span>}
            </button>
          ))}
        </div>

        {error && <div className="graph-error">{error}</div>}
        {nodes === null && !error && <div className="graph-loading">Loading…</div>}
        {nodes !== null && nodes.length === 0 && !error && (
          <div className="graph-empty">No {noun} meet the minimum word count yet.</div>
        )}

        {nodes !== null && nodes.length > 0 && (
          <div className="domain-map-canvas" ref={wrapRef}>
            <canvas
              ref={canvasRef}
              role="img"
              aria-label={`Map of ${nodes.length} ${noun} by discipline category`}
              onMouseMove={onMove}
              onMouseLeave={() => setHovered(null)}
              onClick={onClick}
              style={{ cursor: hovered ? 'pointer' : 'default' }}
            />
            {hovered && (
              <div className="domain-map-tooltip" style={{ left: hovered.x + 12, top: hovered.y + 12 }}>
                <strong>{hovered.p.label}</strong>
                {hovered.p.subtitle && <span className="muted"> · {hovered.p.subtitle}</span>}
                <span>
                  {hovered.p.dominant_name} ({Math.round(hovered.p.dominant_fraction * 100)}%)
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default DomainMap
