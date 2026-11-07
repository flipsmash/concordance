import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { colorForCluster } from './clusterColors'
import { ZOOM_MS, cssVar } from './graphUtils'
import SharedWordsPanel from './SharedWordsPanel'
import './CorpusMap.css'
import './GraphView.css' // .graph-maximize, shared with the other maps' fullscreen button

const API_BASE = ''
const WORLD = 1000 // corpus_map.COORD_SCALE: stored coordinates are 0..1000
const PAD = 24
const SUB_LABELS_AT = 2.5 // zoom (relative to fit) where sub-cluster labels replace cluster labels
const TITLES_AT = 7 // ... and where individual titles appear
const MAX_TITLES = 220
const HIT_PX = 7

// The full-corpus map: every book (or author) placed by `concordance
// corpus-map`, which the other tabs' top-200 views can't show. Canvas, not
// SVG -- ~29k points. Zoomed out it shows clusters labelled by their most
// distinctive words; zooming in brings in sub-cluster labels, then titles.
// Positions are approximate (2-D can't keep every neighbourhood), so the
// side panel's neighbour list -- read from book_similarity /
// author_similarity -- is the exact answer and says so.
function CorpusMap({ entity, onOpen, relatedUrl, sharedWordsUrl }) {
  const [data, setData] = useState(null) // null = loading
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(null) // point index
  const [focusCluster, setFocusCluster] = useState(null) // top-level cluster id, dims the rest
  const [expanded, setExpanded] = useState(() => new Set())
  const [query, setQuery] = useState('')
  const [hover, setHover] = useState(null) // {i, x, y} in canvas px
  const [neighbours, setNeighbours] = useState(null)
  const [shared, setShared] = useState(null) // {key, label}
  const [, setRedraw] = useState(0)

  const rootRef = useRef(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const wrapRef = useRef(null)
  const canvasRef = useRef(null)
  const view = useRef({ k: 1, tx: 0, ty: 0, fitK: 1 }) // screen = world * k + t
  const size = useRef({ w: 0, h: 0 })
  const anim = useRef(null)
  const frame = useRef(0)

  useEffect(() => {
    setData(null)
    setError('')
    setSelected(null)
    setFocusCluster(null)
    fetch(`${API_BASE}/api/browse/corpus-map?entity=${entity}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then(setData)
      .catch((err) => setError(err.message || 'failed to load the map'))
  }, [entity])

  // Maximize = browser fullscreen on the whole map + panel, like the other
  // visualizations; the canvas's ResizeObserver redraws at the new size.
  useEffect(() => {
    const onChange = () => setIsFullscreen(document.fullscreenElement === rootRef.current)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen()
    else rootRef.current?.requestFullscreen()
  }

  const pts = data?.points
  const n = pts ? pts.key.length : 0

  // Spatial grid over world coordinates for hover/click hit-testing.
  const grid = useMemo(() => {
    if (!pts) return null
    const cell = 8
    const cells = new Map()
    for (let i = 0; i < n; i++) {
      const id = `${Math.floor(pts.x[i] / cell)},${Math.floor(pts.y[i] / cell)}`
      let list = cells.get(id)
      if (!list) cells.set(id, (list = []))
      list.push(i)
    }
    return { cell, cells }
  }, [pts, n])

  const byCluster = useMemo(() => {
    const m = new Map()
    if (!pts) return m
    for (let i = 0; i < n; i++) {
      let list = m.get(pts.cluster[i])
      if (!list) m.set(pts.cluster[i], (list = []))
      list.push(i)
    }
    return m
  }, [pts, n])

  const tree = useMemo(() => {
    if (!data) return []
    const top = data.clusters.filter((c) => c.subcluster === -1).sort((a, b) => b.size - a.size)
    return top.map((c) => ({
      ...c,
      subs: data.clusters.filter((s) => s.cluster === c.cluster && s.subcluster >= 0).sort((a, b) => b.size - a.size),
    }))
  }, [data])

  // Title order when zoomed in: books with the most distinctive words first,
  // so the labels that win a crowded spot are the substantial works.
  const titleOrder = useMemo(() => {
    if (!pts) return []
    return Array.from({ length: n }, (_, i) => i).sort((a, b) => pts.word_count[b] - pts.word_count[a])
  }, [pts, n])

  const keyIndex = useMemo(() => {
    const m = new Map()
    if (pts) for (let i = 0; i < n; i++) m.set(String(pts.key[i]), i)
    return m
  }, [pts, n])

  // --- drawing -------------------------------------------------------------

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !pts) return
    const { w, h } = size.current
    const dpr = window.devicePixelRatio || 1
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = cssVar('--bg', '#fff')
    ctx.fillRect(0, 0, w, h)
    const { k, tx, ty, fitK } = view.current
    const zoom = k / fitK
    const r = Math.min(4, 1.1 + Math.sqrt(zoom) * 0.6)

    for (const [c, idxs] of byCluster) {
      const dim = focusCluster !== null && c !== focusCluster
      ctx.globalAlpha = dim ? 0.08 : 0.75
      ctx.fillStyle = colorForCluster(c)
      for (const i of idxs) {
        const sx = pts.x[i] * k + tx
        const sy = pts.y[i] * k + ty
        if (sx < -4 || sy < -4 || sx > w + 4 || sy > h + 4) continue
        ctx.fillRect(sx - r / 2, sy - r / 2, r, r)
      }
    }
    ctx.globalAlpha = 1

    const textColor = cssVar('--text-h', '#10181f')
    const haloColor = cssVar('--bg', '#fff')
    const labelAt = (text, sx, sy, font) => {
      ctx.font = font
      ctx.lineWidth = 3
      ctx.strokeStyle = haloColor
      ctx.strokeText(text, sx, sy)
      ctx.fillStyle = textColor
      ctx.fillText(text, sx, sy)
    }
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'

    if (zoom < TITLES_AT) {
      const level = zoom < SUB_LABELS_AT ? -1 : 0
      const rows = data.clusters.filter((c) => (level === -1 ? c.subcluster === -1 : c.subcluster >= 0))
      const taken = []
      for (const c of [...rows].sort((a, b) => b.size - a.size)) {
        if (focusCluster !== null && c.cluster !== focusCluster) continue
        const sx = c.x * k + tx
        const sy = c.y * k + ty
        if (sx < 0 || sy < 0 || sx > w || sy > h) continue
        const text = c.label_words.slice(0, level === -1 ? 3 : 2).join(' · ')
        if (!text) continue
        const font = `${level === -1 ? 600 : 500} 12px system-ui, sans-serif`
        ctx.font = font
        const half = ctx.measureText(text).width / 2
        const lx = Math.min(w - half - 4, Math.max(half + 4, sx)) // keep it inside the canvas
        if (taken.some((t) => Math.abs(t.x - lx) < half + t.half + 6 && Math.abs(t.y - sy) < 16)) continue
        taken.push({ x: lx, y: sy, half })
        labelAt(text, lx, sy, font)
      }
    } else {
      // Titles of visible points, biggest books first, skipping any whose
      // box would overlap one already drawn.
      const font = '11px system-ui, sans-serif'
      ctx.font = font
      ctx.textAlign = 'left'
      const boxes = []
      for (const i of titleOrder) {
        if (boxes.length >= MAX_TITLES) break
        if (focusCluster !== null && pts.cluster[i] !== focusCluster) continue
        const sx = pts.x[i] * k + tx
        const sy = pts.y[i] * k + ty
        if (sx < 0 || sy < 0 || sx > w || sy > h) continue
        const text = truncate(pts.label[i], 26)
        const box = { x0: sx + 4, x1: sx + 8 + ctx.measureText(text).width, y0: sy - 8, y1: sy + 8 }
        if (boxes.some((b) => box.x0 < b.x1 && b.x0 < box.x1 && box.y0 < b.y1 && b.y0 < box.y1)) continue
        boxes.push(box)
        labelAt(text, sx + 5, sy, font)
      }
    }

    if (selected !== null) {
      const sx = pts.x[selected] * k + tx
      const sy = pts.y[selected] * k + ty
      ctx.beginPath()
      ctx.arc(sx, sy, 7, 0, Math.PI * 2)
      ctx.lineWidth = 2.5
      ctx.strokeStyle = cssVar('--accent', '#1c6dbd')
      ctx.stroke()
      ctx.textAlign = 'left'
      labelAt(truncate(pts.label[selected], 40), sx + 10, sy, '600 12px system-ui, sans-serif')
    }
  }, [pts, data, byCluster, focusCluster, selected, titleOrder])

  const requestDraw = useCallback(() => {
    cancelAnimationFrame(frame.current)
    frame.current = requestAnimationFrame(draw)
  }, [draw])

  useEffect(requestDraw, [requestDraw])
  // For effects that must not re-run whenever the drawing inputs change.
  const requestDrawRef = useRef(requestDraw)
  requestDrawRef.current = requestDraw

  // Light/dark switches redraw with the new palette.
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    mq.addEventListener('change', requestDraw)
    return () => mq.removeEventListener('change', requestDraw)
  }, [requestDraw])

  const fit = useCallback(() => {
    const { w, h } = size.current
    const k = Math.min((w - PAD * 2) / WORLD, (h - PAD * 2) / WORLD)
    view.current = { k, fitK: k, tx: (w - WORLD * k) / 2, ty: (h - WORLD * k) / 2 }
  }, [])

  // Canvas sized to its box (and the device pixel ratio).
  useEffect(() => {
    const wrap = wrapRef.current
    const canvas = canvasRef.current
    if (!wrap || !canvas) return
    let first = true // fit the whole map once per load; later resizes keep the view
    const ro = new ResizeObserver(() => {
      const { width, height } = wrap.getBoundingClientRect()
      size.current = { w: width, h: height }
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      if (first) fit()
      first = false
      requestDrawRef.current()
    })
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [fit, data])

  const animateTo = useCallback(
    (target) => {
      cancelAnimationFrame(anim.current)
      const from = { ...view.current }
      const start = performance.now()
      const step = (now) => {
        const t = Math.min(1, (now - start) / ZOOM_MS)
        const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2
        view.current = {
          ...from,
          k: from.k + (target.k - from.k) * e,
          tx: from.tx + (target.tx - from.tx) * e,
          ty: from.ty + (target.ty - from.ty) * e,
        }
        draw()
        if (t < 1) anim.current = requestAnimationFrame(step)
        else setRedraw((x) => x + 1)
      }
      anim.current = requestAnimationFrame(step)
    },
    [draw],
  )

  const flyToPoint = useCallback(
    (i) => {
      const { w, h } = size.current
      const k = Math.max(view.current.k, view.current.fitK * (TITLES_AT + 1))
      animateTo({ k, tx: w / 2 - pts.x[i] * k, ty: h / 2 - pts.y[i] * k })
    },
    [animateTo, pts],
  )

  const flyToCluster = useCallback(
    (c, sub = null) => {
      const idxs = (byCluster.get(c) || []).filter((i) => sub === null || pts.subcluster[i] === sub)
      if (!idxs.length) return
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
      for (const i of idxs) {
        x0 = Math.min(x0, pts.x[i]); x1 = Math.max(x1, pts.x[i])
        y0 = Math.min(y0, pts.y[i]); y1 = Math.max(y1, pts.y[i])
      }
      const { w, h } = size.current
      const k = Math.min((w - PAD * 4) / Math.max(1, x1 - x0), (h - PAD * 4) / Math.max(1, y1 - y0),
        view.current.fitK * 40)
      animateTo({ k, tx: w / 2 - ((x0 + x1) / 2) * k, ty: h / 2 - ((y0 + y1) / 2) * k })
    },
    [animateTo, byCluster, pts],
  )

  // --- interaction ---------------------------------------------------------

  const hitTest = useCallback(
    (sx, sy) => {
      if (!grid) return null
      const { k, tx, ty } = view.current
      const wx = (sx - tx) / k
      const wy = (sy - ty) / k
      const reach = HIT_PX / k
      const c0 = Math.floor((wx - reach) / grid.cell), c1 = Math.floor((wx + reach) / grid.cell)
      const r0 = Math.floor((wy - reach) / grid.cell), r1 = Math.floor((wy + reach) / grid.cell)
      let best = null, bestD = reach * reach
      for (let cx = c0; cx <= c1; cx++)
        for (let cy = r0; cy <= r1; cy++)
          for (const i of grid.cells.get(`${cx},${cy}`) || []) {
            if (focusCluster !== null && pts.cluster[i] !== focusCluster) continue
            const d = (pts.x[i] - wx) ** 2 + (pts.y[i] - wy) ** 2
            if (d <= bestD) { bestD = d; best = i }
          }
      return best
    },
    [grid, pts, focusCluster],
  )

  const pointers = useRef(new Map())
  const gesture = useRef(null)

  function localXY(e) {
    const rect = canvasRef.current.getBoundingClientRect()
    return [e.clientX - rect.left, e.clientY - rect.top]
  }

  function zoomAt(sx, sy, factor) {
    const v = view.current
    const k = Math.min(v.fitK * 80, Math.max(v.fitK * 0.6, v.k * factor))
    const f = k / v.k
    view.current = { ...v, k, tx: sx - (sx - v.tx) * f, ty: sy - (sy - v.ty) * f }
    requestDraw()
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    // Non-passive, so the page doesn't scroll while zooming the map.
    const onWheel = (e) => {
      e.preventDefault()
      const [sx, sy] = localXY(e)
      zoomAt(sx, sy, Math.exp(-e.deltaY * 0.0015))
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  })

  function onPointerDown(e) {
    canvasRef.current.setPointerCapture(e.pointerId)
    pointers.current.set(e.pointerId, localXY(e))
    const ps = [...pointers.current.values()]
    gesture.current = ps.length === 1
      ? { kind: 'pan', start: ps[0], moved: false, view: { ...view.current } }
      : { kind: 'pinch', dist: Math.hypot(ps[0][0] - ps[1][0], ps[0][1] - ps[1][1]), view: { ...view.current } }
  }

  function onPointerMove(e) {
    const xy = localXY(e)
    if (!pointers.current.has(e.pointerId)) {
      const i = hitTest(...xy)
      setHover(i === null ? null : { i, x: xy[0], y: xy[1] })
      return
    }
    pointers.current.set(e.pointerId, xy)
    const g = gesture.current
    const ps = [...pointers.current.values()]
    if (g?.kind === 'pan' && ps.length === 1) {
      const dx = xy[0] - g.start[0], dy = xy[1] - g.start[1]
      if (Math.abs(dx) + Math.abs(dy) > 3) g.moved = true
      view.current = { ...g.view, tx: g.view.tx + dx, ty: g.view.ty + dy }
      setHover(null)
      requestDraw()
    } else if (g?.kind === 'pinch' && ps.length === 2) {
      const d = Math.hypot(ps[0][0] - ps[1][0], ps[0][1] - ps[1][1])
      const mx = (ps[0][0] + ps[1][0]) / 2, my = (ps[0][1] + ps[1][1]) / 2
      view.current = { ...g.view }
      zoomAt(mx, my, d / g.dist)
    }
  }

  function onPointerUp(e) {
    const g = gesture.current
    pointers.current.delete(e.pointerId)
    if (g?.kind === 'pan' && !g.moved && pointers.current.size === 0) {
      const i = hitTest(...localXY(e))
      setSelected(i)
    }
    if (pointers.current.size === 0) {
      gesture.current = null
      setRedraw((x) => x + 1)
    }
  }

  // --- selection: exact neighbours from book_similarity / author_similarity

  useEffect(() => {
    setNeighbours(null)
    if (selected === null || !pts) return
    let live = true
    fetch(`${API_BASE}${relatedUrl(pts.key[selected])}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((d) => {
        if (!live) return
        if (!d) return setNeighbours([])
        const center = String(d.center.id)
        const edge = new Map(d.edges.filter((ed) => ed.is_center_edge)
          .map((ed) => [String(String(ed.source) === center ? ed.target : ed.source), ed]))
        setNeighbours(d.nodes.filter((nd) => nd.ring === 1)
          .map((nd) => ({ key: String(nd.id), label: nd.title ?? nd.id, sub: nd.author,
            score: edge.get(String(nd.id))?.score ?? 0, shared: edge.get(String(nd.id))?.shared_word_count ?? 0 }))
          .sort((a, b) => b.score - a.score))
      })
      .catch(() => live && setNeighbours([]))
    return () => { live = false }
  }, [selected, pts, relatedUrl])

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q || !pts) return []
    const out = []
    for (let i = 0; i < n && out.length < 25; i++) {
      if (pts.label[i].toLowerCase().includes(q) || (pts.author && pts.author[i].toLowerCase().includes(q))) out.push(i)
    }
    return out
  }, [query, pts, n])

  function choose(i) {
    setFocusCluster(null)
    setSelected(i)
    setQuery('')
    flyToPoint(i)
  }

  // --- render --------------------------------------------------------------

  if (error) return <div className="error-banner">{error}</div>
  if (!data) return <p className="muted">Loading the map…</p>
  if (!data.computed_at) {
    return (
      <p className="muted corpus-map-empty">
        The full map hasn't been computed yet. Run <code>concordance corpus-map</code>.
      </p>
    )
  }

  const noun = entity === 'book' ? 'books' : 'authors'
  const sel = selected
  const selCluster = sel !== null ? tree.find((c) => c.cluster === pts.cluster[sel]) : null
  const selSub = selCluster?.subs.find((s) => s.subcluster === pts.subcluster[sel])

  return (
    <div className="corpus-map" ref={rootRef}>
      <div className="corpus-map-top">
        <p className="muted corpus-map-note">
          All {data.n.toLocaleString()} {noun}, mapped {new Date(data.computed_at).toLocaleDateString()}. Position is
          approximate; the neighbour list is exact. {entity === 'book' && 'Books added since then aren’t on the map yet.'}
        </p>
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>
      <div className="corpus-map-body">
        <div className="corpus-map-canvas" ref={wrapRef}>
          <canvas
            ref={canvasRef}
            role="img"
            aria-label={`Map of all ${data.n} ${noun}, grouped into ${tree.length} clusters`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onPointerLeave={() => setHover(null)}
            onDoubleClick={(e) => zoomAt(...localXY(e), 2)}
          />
          {hover && hover.i !== selected && (
            <div className="corpus-map-tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
              <strong>{pts.label[hover.i]}</strong>
              {pts.author && pts.author[hover.i] && <span className="muted"> — {pts.author[hover.i]}</span>}
            </div>
          )}
          <div className="corpus-map-zoom">
            <button type="button" onClick={() => zoomAt(size.current.w / 2, size.current.h / 2, 1.6)} aria-label="Zoom in">+</button>
            <button type="button" onClick={() => zoomAt(size.current.w / 2, size.current.h / 2, 1 / 1.6)} aria-label="Zoom out">−</button>
            <button type="button" onClick={() => { setFocusCluster(null); fit(); requestDraw(); setRedraw((x) => x + 1) }}>
              Reset
            </button>
          </div>
        </div>

        <aside className="corpus-map-panel">
          <div className="corpus-map-search">
            <input
              type="search"
              value={query}
              placeholder={entity === 'book' ? 'Find a book or author…' : 'Find an author…'}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && matches.length && choose(matches[0])}
              aria-label="Search the map"
            />
            {query.trim() && (
              <ul className="corpus-map-matches">
                {matches.map((i) => (
                  <li key={i}>
                    <button type="button" onClick={() => choose(i)}>
                      {pts.label[i]}
                      {pts.author && pts.author[i] && <span className="muted"> — {pts.author[i]}</span>}
                    </button>
                  </li>
                ))}
                {!matches.length && (
                  <li className="muted corpus-map-nomatch">
                    Not on the map{entity === 'book' ? ' — it may have been added since the last run' : ''}.
                  </li>
                )}
              </ul>
            )}
          </div>

          {sel !== null ? (
            <div className="corpus-map-detail">
              <button type="button" className="corpus-map-back" onClick={() => setSelected(null)}>← Clusters</button>
              <h3>{pts.label[sel]}</h3>
              {pts.author && pts.author[sel] && <p className="muted">{pts.author[sel]}</p>}
              {selCluster && (
                <p className="corpus-map-in">
                  <span className="corpus-map-swatch" style={{ background: colorForCluster(selCluster.cluster) }} />
                  {(selSub?.label_words.length ? selSub : selCluster).label_words.slice(0, 5).join(', ')}
                </p>
              )}
              {pts.sparse[sel] === 1 && (
                <p className="muted corpus-map-sparse">
                  Only {pts.word_count[sel]} distinctive words, so this position is a rough guess.
                </p>
              )}
              <button type="button" className="corpus-map-open" onClick={() => onOpen(pts, sel)}>
                Open {entity === 'book' ? 'book' : 'author'} page →
              </button>
              <h4>Nearest {noun}</h4>
              {neighbours === null && <p className="muted">Loading…</p>}
              {neighbours && !neighbours.length && <p className="muted">No stored neighbours.</p>}
              {neighbours && neighbours.length > 0 && (
                <ol className="corpus-map-neighbours">
                  {neighbours.map((nb) => {
                    const idx = keyIndex.get(nb.key)
                    return (
                      <li key={nb.key}>
                        <button type="button" disabled={idx === undefined} onClick={() => idx !== undefined && choose(idx)}
                          title={idx === undefined ? 'Not on the map yet' : 'Show on the map'}>
                          {nb.label}
                          {nb.sub && <span className="muted"> — {nb.sub}</span>}
                        </button>
                        <button type="button" className="corpus-map-shared"
                          onClick={() => setShared({ key: nb.key, label: nb.label })}>
                          {nb.shared} shared
                        </button>
                      </li>
                    )
                  })}
                </ol>
              )}
            </div>
          ) : (
            <div className="corpus-map-clusters">
              <p className="muted">
                {tree.length} clusters, named by their most distinctive words. Click one to zoom in.
              </p>
              <ul>
                {tree.map((c) => (
                  <li key={c.cluster} className={focusCluster === c.cluster ? 'active' : ''}>
                    <div className="corpus-map-cluster-row">
                      <button type="button" className="corpus-map-expand" aria-expanded={expanded.has(c.cluster)}
                        aria-label={expanded.has(c.cluster) ? 'Collapse' : 'Expand'}
                        onClick={() => setExpanded((s) => toggle(s, c.cluster))}>
                        {expanded.has(c.cluster) ? '▾' : '▸'}
                      </button>
                      <button type="button" className="corpus-map-cluster-name"
                        onClick={() => {
                          const on = focusCluster === c.cluster
                          setFocusCluster(on ? null : c.cluster)
                          if (!on) flyToCluster(c.cluster)
                        }}>
                        <span className="corpus-map-swatch" style={{ background: colorForCluster(c.cluster) }} />
                        <span className="corpus-map-words">{c.label_words.slice(0, 4).join(', ') || 'unlabelled'}</span>
                        <span className="muted corpus-map-size">{c.size.toLocaleString()}</span>
                      </button>
                    </div>
                    {expanded.has(c.cluster) && (
                      <ul className="corpus-map-subs">
                        {c.subs.map((s) => (
                          <li key={s.subcluster}>
                            <button type="button" className="corpus-map-cluster-name"
                              onClick={() => { setFocusCluster(c.cluster); flyToCluster(c.cluster, s.subcluster) }}>
                              <span className="corpus-map-words">{s.label_words.slice(0, 4).join(', ') || 'unlabelled'}</span>
                              <span className="muted corpus-map-size">{s.size.toLocaleString()}</span>
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </aside>
      </div>
      {shared && sel !== null && (
        <SharedWordsPanel
          fetchUrl={sharedWordsUrl(pts.key[sel], shared.key)}
          titleA={pts.label[sel]}
          titleB={shared.label}
          onClose={() => setShared(null)}
        />
      )}
    </div>
  )
}

function truncate(s, max) {
  return s.length > max ? `${s.slice(0, max - 1)}…` : s
}

function toggle(set, v) {
  const next = new Set(set)
  if (next.has(v)) next.delete(v)
  else next.add(v)
  return next
}

export default CorpusMap
