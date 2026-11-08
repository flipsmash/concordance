import { useEffect, useMemo, useRef, useState } from 'react'
import { isDark } from './domainColors'
import './ClusterMap.css'
import './GraphView.css' // .graph-controls / .graph-maximize / .graph-loading / .graph-empty / .graph-error

const API_BASE = ''
const VIEW = 600
const PADDING = 44
const R_MIN = 4
const R_MAX = 13
// Validated default categorical slot 1 (light / dark), the one hue this map
// uses: clusters are identified by their NAMES, not by colour -- twelve
// clusters is past the eight-hue ceiling, and in a scatter even eight can't
// all be told apart pairwise.
const HUE = { light: '#2a78d6', dark: '#3987e5' }

function normalise(entity, nodes) {
  return nodes.map((n) =>
    entity === 'book'
      ? { key: n.id, label: n.title, sub: n.author, cluster: n.cluster_id, x: n.x, y: n.y, weight: n.word_count,
          weightText: `${n.word_count.toLocaleString()} words`, raw: n }
      : { key: n.author, label: n.author, sub: null, cluster: n.cluster_id, x: n.x, y: n.y, weight: n.book_count,
          weightText: `${n.book_count} book${n.book_count === 1 ? '' : 's'}`, raw: n },
  )
}

// The top-N relatedness map (one clustering run, positioned by MDS). Each
// cluster is named by its most distinctive words (/api/browse/clusters) and
// labelled where it sits; hovering a dot, a label or a legend row lights up
// that whole cluster.
function ClusterMap({ entity, scope = 'volume', highlightKey = null, onItemClick }) {
  const [items, setItems] = useState(null) // null = loading
  const [clusters, setClusters] = useState([])
  const [error, setError] = useState('')
  const [hoverItem, setHoverItem] = useState(null)
  const [hoverCluster, setHoverCluster] = useState(null)
  const [pinned, setPinned] = useState(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const containerRef = useRef(null)
  const noun = entity === 'book' ? 'books' : 'authors'

  useEffect(() => {
    setItems(null)
    setError('')
    setPinned(null)
    Promise.all([
      fetch(`${API_BASE}/api/browse/${noun}/map?scope=${scope}`).then((r) => {
        if (!r.ok) throw new Error(`request failed (${r.status})`)
        return r.json()
      }),
      fetch(`${API_BASE}/api/browse/clusters?entity=${entity}&scope=${scope}`).then((r) => (r.ok ? r.json() : { clusters: [] })),
    ])
      .then(([map, cl]) => {
        setItems(normalise(entity, map.nodes))
        setClusters(cl.clusters)
      })
      .catch((err) => setError(err.message || 'failed to load the map'))
  }, [entity, noun, scope])

  useEffect(() => {
    const onChange = () => setIsFullscreen(document.fullscreenElement === containerRef.current)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen()
    else containerRef.current?.requestFullscreen()
  }

  const { points, centroids } = useMemo(() => {
    if (!items || !items.length) return { points: [], centroids: [] }
    const xs = items.map((n) => n.x), ys = items.map((n) => n.y)
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys)
    const spanX = maxX - minX || 1, spanY = maxY - minY || 1
    const scale = (VIEW - 2 * PADDING) / Math.max(spanX, spanY)
    const maxW = Math.max(...items.map((n) => n.weight), 1)
    const pts = items.map((n) => ({
      ...n,
      cx: PADDING + (n.x - minX) * scale + (VIEW - 2 * PADDING - spanX * scale) / 2,
      cy: PADDING + (maxY - n.y) * scale + (VIEW - 2 * PADDING - spanY * scale) / 2,
      r: R_MIN + Math.sqrt(n.weight / maxW) * (R_MAX - R_MIN), // area-proportional
    }))
    const byC = new Map()
    for (const p of pts) {
      const c = byC.get(p.cluster) || { x: 0, y: 0, n: 0 }
      c.x += p.cx; c.y += p.cy; c.n += 1
      byC.set(p.cluster, c)
    }
    const cents = [...byC.entries()].map(([cluster, c]) => ({ cluster, x: c.x / c.n, y: c.y / c.n, n: c.n }))
    return { points: pts, centroids: cents }
  }, [items])

  const nameOf = useMemo(() => {
    const m = new Map()
    for (const c of clusters) m.set(c.cluster_id, c.label_words)
    return m
  }, [clusters])

  // Cluster labels at their centroids, biggest first, skipping any that
  // would overlap one already placed.
  const labels = useMemo(() => {
    const placed = []
    for (const c of [...centroids].sort((a, b) => b.n - a.n)) {
      const words = (nameOf.get(c.cluster) || []).slice(0, 2)
      if (!words.length) continue
      const text = words.join(' · ')
      const half = (text.length * 6.4) / 2 + 4
      if (placed.some((p) => Math.abs(p.x - c.x) < half + p.half && Math.abs(p.y - c.y) < 16)) continue
      placed.push({ ...c, text, half })
    }
    return placed
  }, [centroids, nameOf])

  const activeCluster =
    hoverCluster ?? (hoverItem ? hoverItem.cluster : null) ?? pinned ??
    (highlightKey != null ? points.find((p) => p.key === highlightKey)?.cluster ?? null : null)
  const hue = HUE[isDark() ? 'dark' : 'light']
  const legend = useMemo(() => [...clusters].sort((a, b) => b.size - a.size), [clusters])

  return (
    <div className="cluster-map" ref={containerRef}>
      <div className="graph-controls cluster-map-controls">
        <span className="muted cluster-map-hint">
          Each group is named by the words that set it apart. Hover a dot or a name to see its group.
        </span>
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>
      {error && <div className="graph-error">{error}</div>}
      {items === null && !error && <div className="graph-loading">Loading…</div>}
      {items !== null && items.length === 0 && !error && (
        <div className="graph-empty">
          No cluster data yet -- run `concordance {entity}-clustering{scope === 'fame' ? ' --min-fame 8' : ''}`.
        </div>
      )}
      {points.length > 0 && (
        <div className="cluster-map-body">
          <div className="cluster-map-plot">
            <svg viewBox={`0 0 ${VIEW} ${VIEW}`} className="cluster-map-svg" role="img"
              aria-label={`Map of the top ${points.length} ${noun}, in ${centroids.length} named groups`}>
              {points.map((p) => {
                const on = activeCluster === null || p.cluster === activeCluster
                const isKey = p.key === highlightKey || (hoverItem && p.key === hoverItem.key)
                return (
                  <circle key={p.key} cx={p.cx} cy={p.cy} r={p.r}
                    className={`cluster-map-dot${isKey ? ' key' : ''}`}
                    style={{ fill: hue, opacity: on ? 0.85 : 0.14 }}
                    onMouseEnter={() => setHoverItem(p)}
                    onMouseLeave={() => setHoverItem((h) => (h && h.key === p.key ? null : h))}
                    onClick={() => onItemClick?.(p.raw)} />
                )
              })}
              {labels.map((l) => (
                <text key={l.cluster} x={l.x} y={l.y} textAnchor="middle" dominantBaseline="middle"
                  className={`cluster-map-name${activeCluster === l.cluster ? ' on' : ''}${activeCluster !== null && activeCluster !== l.cluster ? ' off' : ''}`}
                  onMouseEnter={() => setHoverCluster(l.cluster)} onMouseLeave={() => setHoverCluster(null)}>
                  {l.text}
                </text>
              ))}
              {hoverItem && (
                <text x={hoverItem.cx} y={hoverItem.cy - hoverItem.r - 6} textAnchor="middle" className="cluster-map-item-label">
                  {hoverItem.label}{hoverItem.sub ? ` · ${hoverItem.sub}` : ''} · {hoverItem.weightText}
                </text>
              )}
            </svg>
          </div>
          <ol className="cluster-map-legend" aria-label="Groups">
            {legend.map((c) => (
              <li key={c.cluster_id}>
                <button type="button"
                  className={`cluster-map-legend-row${activeCluster === c.cluster_id ? ' on' : ''}`}
                  aria-pressed={pinned === c.cluster_id}
                  onMouseEnter={() => setHoverCluster(c.cluster_id)} onMouseLeave={() => setHoverCluster(null)}
                  onClick={() => setPinned(pinned === c.cluster_id ? null : c.cluster_id)}>
                  <span className="cluster-map-legend-words">{c.label_words.join(', ') || '—'}</span>
                  <span className="muted">{c.size}</span>
                </button>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}

export default ClusterMap
