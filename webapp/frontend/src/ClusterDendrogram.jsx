import { useEffect, useMemo, useRef, useState } from 'react'
import './ClusterDendrogram.css'
import './GraphView.css' // .graph-controls / .graph-maximize / .graph-loading / .graph-empty / .graph-error

const API_BASE = ''
const ROW = 22 // px per leaf row
const TREE_W = 220 // px of branches left of the titles

// The clustering run's linkage tree, turned sideways so every leaf can carry
// its title (200 vertical leaves had room for nothing but a coloured dot).
// Branch depth = merge height, as before; each named group (see
// /api/browse/clusters) is a shaded band with its name on its first row.
function ClusterDendrogram({ entity, scope = 'volume', highlightKey = null, onItemClick }) {
  const [data, setData] = useState(null) // {tree, clusters} | null = loading
  const [error, setError] = useState('')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const containerRef = useRef(null)
  const scrollRef = useRef(null)
  const noun = entity === 'book' ? 'books' : 'authors'

  useEffect(() => {
    setData(null)
    setError('')
    Promise.all([
      fetch(`${API_BASE}/api/browse/${noun}/dendrogram?scope=${scope}`).then((r) => {
        if (!r.ok) throw new Error(`request failed (${r.status})`)
        return r.json()
      }),
      fetch(`${API_BASE}/api/browse/clusters?entity=${entity}&scope=${scope}`).then((r) => (r.ok ? r.json() : { clusters: [] })),
    ])
      .then(([d, cl]) => setData({ tree: d.tree, clusters: cl.clusters }))
      .catch((err) => setError(err.message || 'failed to load the dendrogram'))
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

  const layout = useMemo(() => {
    if (!data?.tree) return null
    const clusterOf = new Map()
    const nameOf = new Map()
    for (const c of data.clusters) {
      nameOf.set(c.cluster_id, c.label_words)
      for (const m of c.members) clusterOf.set(m, c.cluster_id)
    }
    const leaves = []
    const links = []
    const maxD = data.tree.distance || 1
    const xOf = (n) => (n.distance == null ? TREE_W : TREE_W * (1 - n.distance / maxD))
    // returns the node's y; left-first so leaves come out in the run's order
    function walk(n) {
      if (!n.left && !n.right) {
        const key = entity === 'book' ? n.id : n.author
        leaves.push({ key, label: entity === 'book' ? n.title : n.author, sub: entity === 'book' ? n.author : null,
          cluster: clusterOf.get(key), raw: entity === 'book' ? { id: n.id, title: n.title, author: n.author } : n.author })
        return (leaves.length - 0.5) * ROW
      }
      const x = xOf(n)
      const kids = [n.left, n.right].filter(Boolean).map((k) => ({ y: walk(k), x: xOf(k) }))
      const y = kids.reduce((s, k) => s + k.y, 0) / kids.length
      links.push({ x1: x, y1: Math.min(...kids.map((k) => k.y)), x2: x, y2: Math.max(...kids.map((k) => k.y)) })
      for (const k of kids) links.push({ x1: x, y1: k.y, x2: k.x, y2: k.y })
      return y
    }
    walk(data.tree)
    // contiguous runs of one cluster -> bands
    const bands = []
    leaves.forEach((l, i) => {
      const last = bands[bands.length - 1]
      if (last && last.cluster === l.cluster) last.end = i
      else bands.push({ cluster: l.cluster, start: i, end: i, words: nameOf.get(l.cluster) || [] })
    })
    return { leaves, links, bands }
  }, [data, entity])

  useEffect(() => {
    if (!layout || highlightKey == null || !scrollRef.current) return
    const i = layout.leaves.findIndex((l) => l.key === highlightKey)
    if (i >= 0) scrollRef.current.scrollTo({ top: Math.max(0, i * ROW - 120), behavior: 'smooth' })
  }, [layout, highlightKey])

  return (
    <div className="cluster-dendro" ref={containerRef}>
      <div className="graph-controls cluster-dendro-controls">
        <span className="muted cluster-dendro-hint">
          Branches join where {noun} share vocabulary; the further left they join, the less alike. Shaded groups are
          named by the words that set them apart.
        </span>
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>
      {error && <div className="graph-error">{error}</div>}
      {data === null && !error && <div className="graph-loading">Loading…</div>}
      {data && !data.tree && (
        <div className="graph-empty">
          No cluster data yet -- run `concordance {entity}-clustering{scope === 'fame' ? ' --min-fame 8' : ''}`.
        </div>
      )}
      {layout && (
        <div className="cluster-dendro-scroll" ref={scrollRef}>
          <div className="cluster-dendro-grid" style={{ height: layout.leaves.length * ROW }}>
            {layout.bands.map((b, i) => (
              <div key={`${b.cluster}-${b.start}`} className={`cluster-dendro-band${i % 2 ? ' alt' : ''}`}
                style={{ top: b.start * ROW, height: (b.end - b.start + 1) * ROW }}>
                <span className="cluster-dendro-band-name">{b.words.slice(0, 3).join(' · ')}</span>
              </div>
            ))}
            <svg className="cluster-dendro-tree" width={TREE_W} height={layout.leaves.length * ROW} aria-hidden="true">
              {layout.links.map((l, i) => (
                <line key={i} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} />
              ))}
            </svg>
            <ol className="cluster-dendro-leaves" aria-label={`${noun} in tree order`}>
              {layout.leaves.map((l) => (
                <li key={l.key} style={{ height: ROW }}>
                  <button type="button" className={`cluster-dendro-leaf${l.key === highlightKey ? ' on' : ''}`}
                    onClick={() => onItemClick?.(l.raw)} title={l.sub ? `${l.label} — ${l.sub}` : l.label}>
                    {l.label}
                    {l.sub && <span className="muted"> · {l.sub}</span>}
                  </button>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}
    </div>
  )
}

export default ClusterDendrogram
