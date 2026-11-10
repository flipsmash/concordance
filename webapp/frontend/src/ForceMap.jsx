import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Graph from 'graphology'
import Sigma from 'sigma'
import { cssVar } from './graphUtils'
import './GraphView.css' // .graph-controls / .graph-maximize / .graph-signal-toggle
import './ForceMap.css'

// The whole collection as a force-directed map (books, authors or words),
// laid out ahead of time by `concordance force-map` and drawn here with
// sigma.js (WebGL), so tens of thousands of points pan and zoom smoothly.
// Zoomed out you see regions -- communities named by their distinctive
// vocabulary (books/authors) or best-connected words -- and zooming in brings
// up individual labels. One hue for every point (dataviz: an all-pairs
// scatter can't carry many distinguishable colours); picking a region
// highlights it and greys the rest.

const API_BASE = ''
const NOUNS = { book: 'books', author: 'authors', word: 'words' }

function hrefFor(entity, key, label, extra) {
  if (entity === 'word') return `/app/words/${key}`
  if (entity === 'author') return `/app/authors/${encodeURIComponent(key)}`
  return extra ? `/app/authors/${encodeURIComponent(extra)}/${key}` : `/app/books?q=${encodeURIComponent(label)}`
}

function ForceMap({ entity }) {
  const navigate = useNavigate()
  const containerRef = useRef(null)
  const viewRef = useRef(null)
  const sigmaRef = useRef(null)
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [region, setRegion] = useState('')
  const [query, setQuery] = useState('')
  const [found, setFound] = useState(null)
  // the term whose links are shown: hovered, else the one clicked
  const [hovered, setHovered] = useState(null)
  const [selected, setSelected] = useState(null)
  const [showLinks, setShowLinks] = useState(false)
  const [regionLabels, setRegionLabels] = useState([])
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [theme, setTheme] = useState(0) // bumps on a light/dark switch, to recolour

  useEffect(() => {
    setData(null)
    setError('')
    fetch(`${API_BASE}/api/browse/force-map?entity=${entity}`)
      .then((res) => (res.ok ? res.json() : res.json().then((b) => Promise.reject(new Error(b.detail || res.status)))))
      .then(setData)
      .catch((err) => setError(err.message || 'failed to load the map'))
  }, [entity])

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const bump = () => setTheme((t) => t + 1)
    mq.addEventListener('change', bump)
    const obs = new MutationObserver(bump)
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => { mq.removeEventListener('change', bump); obs.disconnect() }
  }, [])

  useEffect(() => {
    function onChange() {
      setIsFullscreen(document.fullscreenElement === viewRef.current)
      // re-centre after entering/leaving fullscreen (Brian's rule for every viz)
      setTimeout(() => sigmaRef.current?.getCamera().animatedReset({ duration: 300 }), 150)
    }
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  const named = useMemo(
    () => (data ? data.regions.filter((r) => r.name).sort((a, b) => b.size - a.size) : []),
    [data],
  )

  // build the graph + renderer once per dataset
  useEffect(() => {
    if (!data || !containerRef.current) return undefined
    const g = new Graph({ type: 'undirected', multi: false })
    const n = data.nodes
    for (let i = 0; i < n.key.length; i++) {
      g.addNode(String(i), { x: n.x[i], y: n.y[i], size: entity === 'word' ? 3 : 3.5, label: n.label[i], region: n.region[i] })
    }
    for (let i = 0; i < data.edges.a.length; i++) {
      const a = String(data.edges.a[i])
      const b = String(data.edges.b[i])
      if (a !== b && !g.hasEdge(a, b)) g.addEdge(a, b, { size: 0.4 })
    }
    let renderer
    try {
      renderer = new Sigma(g, containerRef.current, {
      hideEdgesOnMove: true,
      labelRenderedSizeThreshold: 7,
      labelDensity: 0.6,
      labelGridCellSize: 90,
      zIndex: true,
      minCameraRatio: 0.005,
      maxCameraRatio: 3,
      })
    } catch (err) {
      // sigma needs WebGL 2; without it, say so rather than take the page down
      setError(/webgl/i.test(String(err?.message))
        ? 'This map needs WebGL 2, which this browser isn’t providing (try another browser, or turn on hardware acceleration).'
        : `The map couldn’t be drawn: ${err?.message || err}`)
      return undefined
    }
    // positions carry the relations; links only appear for the term in focus
    renderer.on('enterNode', ({ node }) => setHovered(node))
    renderer.on('leaveNode', () => setHovered(null))
    renderer.on('clickNode', ({ node }) => setSelected((cur) => (cur === node ? null : node)))
    renderer.on('clickStage', () => setSelected(null))
    renderer.on('afterRender', () => {
      const ratio = renderer.getCamera().ratio
      const show = ratio > 0.25
      setRegionLabels(show
        ? named.slice(0, 18).map((r) => ({ ...r, ...renderer.graphToViewport({ x: r.x, y: r.y }) }))
        : [])
    })
    sigmaRef.current = renderer
    return () => {
      renderer.kill()
      sigmaRef.current = null
    }
  }, [data, entity, named])

  // colours + region highlight + search hit, without rebuilding
  useEffect(() => {
    const renderer = sigmaRef.current
    if (!renderer) return
    const hue = cssVar('--accent', '#2a78d6')
    const faint = cssVar('--border', '#d6d4da')
    const text = cssVar('--text-h', '#08060d')
    const pick = region === '' ? null : Number(region)
    const graph = renderer.getGraph()
    const focus = hovered ?? selected ?? found
    const near = focus ? new Set(graph.neighbors(focus)) : null
    renderer.setSetting('labelColor', { color: text })
    renderer.setSetting('defaultEdgeColor', cssVar('--text', '#6b6375'))
    renderer.setSetting('nodeReducer', (node, attrs) => {
      const isFocus = node === focus
      const isNear = near?.has(node)
      const dim = (pick !== null && attrs.region !== pick) || (near && !isFocus && !isNear)
      return {
        ...attrs,
        color: dim ? faint : hue,
        zIndex: isFocus ? 3 : isNear ? 2 : dim ? 0 : 1,
        size: isFocus ? attrs.size * 2.2 : attrs.size,
        forceLabel: isFocus || isNear,
        label: dim && !isNear ? null : attrs.label,
      }
    })
    renderer.setSetting('edgeReducer', (edge, attrs) => {
      const [s, t] = graph.extremities(edge)
      if (focus) return { ...attrs, hidden: s !== focus && t !== focus, size: 1 }
      if (!showLinks) return { ...attrs, hidden: true }
      if (pick === null) return attrs
      const keep = graph.getNodeAttribute(s, 'region') === pick && graph.getNodeAttribute(t, 'region') === pick
      return { ...attrs, hidden: !keep }
    })
    renderer.refresh()
  }, [region, found, theme, data, hovered, selected, showLinks])

  function search(e) {
    e.preventDefault()
    const q = query.trim().toLowerCase()
    if (!q || !data) return
    const labels = data.nodes.label
    let i = labels.findIndex((l) => l.toLowerCase() === q)
    if (i < 0) i = labels.findIndex((l) => l.toLowerCase().startsWith(q))
    if (i < 0) i = labels.findIndex((l) => l.toLowerCase().includes(q))
    if (i < 0) {
      setFound(null)
      setError(`No ${entity} matching “${query.trim()}” on the map`)
      return
    }
    setError('')
    const node = String(i)
    setFound(node)
    setSelected(node)
    setRegion('')
    const renderer = sigmaRef.current
    const pos = renderer?.getNodeDisplayData(node)
    if (pos) renderer.getCamera().animate({ x: pos.x, y: pos.y, ratio: 0.04 }, { duration: 700 })
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen()
    else viewRef.current?.requestFullscreen()
  }

  return (
    <div className="force-map" ref={viewRef}>
      <div className="graph-controls force-map-controls">
        <form onSubmit={search} className="force-map-search">
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={`Find a ${entity}…`}
            aria-label={`Find a ${entity} on the map`} />
        </form>
        {named.length > 0 && (
          <select value={region} onChange={(e) => { setRegion(e.target.value); setFound(null) }}
            aria-label="Highlight a region">
            <option value="">All regions</option>
            {named.map((r) => (
              <option key={r.region} value={r.region}>{r.name} ({r.size.toLocaleString()})</option>
            ))}
          </select>
        )}
        <label className="force-map-links">
          <input type="checkbox" checked={showLinks} onChange={(e) => setShowLinks(e.target.checked)} />
          Show all links
        </label>
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>
      {error && <div className="graph-error">{error}</div>}
      <div className="force-map-canvas-wrap">
        <div className="force-map-canvas" ref={containerRef} />
        {!data && !error && <div className="graph-loading">Loading every {entity}…</div>}
        {selected && data && (
          <div className="force-map-selected">
            <strong>{data.nodes.label[Number(selected)]}</strong>
            <a href={hrefFor(entity, data.nodes.key[Number(selected)], data.nodes.label[Number(selected)], data.nodes.extra?.[Number(selected)])}
              onClick={(e) => { e.preventDefault(); const i = Number(selected); navigate(hrefFor(entity, data.nodes.key[i], data.nodes.label[i], data.nodes.extra?.[i])) }}>
              Open →
            </a>
            <button type="button" aria-label="Close" onClick={() => setSelected(null)}>×</button>
          </div>
        )}
        {regionLabels.map((r) => (
          <span key={r.region} className={`force-map-region${String(r.region) === region ? ' on' : ''}`}
            style={{ left: r.x, top: r.y }}>
            {r.name}
          </span>
        ))}
      </div>
      {data && (
        <p className="muted force-map-caption">
          {data.nodes.key.length.toLocaleString()} {NOUNS[entity]}, each pulled toward the ones it shares the most
          with{entity === 'word' ? ' in meaning' : ' in rare vocabulary'}, so neighbours sit close. Drag to move, scroll
          or pinch to zoom; names appear as you zoom in. Point at one to see its links; click it to keep them and open it. Laid out {new Date(data.computed_at).toLocaleDateString()}.
        </p>
      )}
    </div>
  )
}

export default ForceMap
