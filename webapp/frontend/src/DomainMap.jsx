import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { colorForBucket } from './domainColors'
import './DomainMap.css'
import './GraphView.css' // .graph-controls/.graph-maximize/.graph-signal-toggle/.graph-error/.graph-loading/.graph-empty

const API_BASE = ''
const VIEW = 600 // SVG viewBox is VIEW x VIEW, coordinates normalized into it
const PADDING = 40
const RADIUS_MIN = 3
const RADIUS_MAX = 13

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
// See browse.py's /api/browse/domain-map for the PCA + corpus-relative-lift
// color computation. Plain SVG, same reasoning as AuthorClusterMap: these
// points are precomputed server-side per request, no client-side simulation
// to run.
function DomainMap() {
  const navigate = useNavigate()
  const [entity, setEntity] = useState('book')
  const [nodes, setNodes] = useState(null) // null = loading
  const [error, setError] = useState('')
  const [hovered, setHovered] = useState(null)
  const [legend, setLegend] = useState([])
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [spread, setSpread] = useState(SPREAD_DEFAULT)
  // Buckets the user has clicked off in the legend -- kept as a set of
  // domain-bucket keys (not entity ids), so it survives switching between
  // the Works/Authors tabs (a filter like "hide Nature & Science" is a
  // property of how you want to look at the map, not of which entity is
  // currently shown).
  const [hiddenBuckets, setHiddenBuckets] = useState(() => new Set())
  const containerRef = useRef(null)

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

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen()
    else containerRef.current?.requestFullscreen()
  }

  const { points, maxWordCount } = useMemo(() => {
    if (!nodes || nodes.length === 0) return { points: [], maxWordCount: 1 }
    const xs = nodes.map((n) => n.x)
    const ys = nodes.map((n) => n.y)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const spanX = maxX - minX || 1
    const spanY = maxY - minY || 1
    const span = Math.max(spanX, spanY)
    // The base scale fits the UNZOOMED extent to the viewBox -- fixed per
    // dataset, independent of spread. spread then multiplies straight onto
    // it as real magnification: a first attempt instead raised each point's
    // distance from the center to a power, which sounded like "spread" but
    // wasn't -- every point still got re-fit to the same box afterward, so
    // a whole dense cluster near the center (which is most points, in a
    // PCA/cosine layout) landed at nearly the same NEW radius once pushed
    // outward, forming a ring around an empty middle instead of spreading
    // out. Multiplying the fixed base scale directly is a true uniform zoom:
    // every pairwise distance grows by the same factor, so a crowded
    // cluster becomes individually visible without any point relative to
    // any OTHER point moving the wrong way. The tradeoff, same as zooming
    // any map, is that far-out points can scroll past the visible edge at
    // high spread -- expected, not a bug.
    const scale = ((VIEW - 2 * PADDING) / span) * spread
    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2
    const maxWords = Math.max(...nodes.map((n) => n.word_count), 1)
    return {
      maxWordCount: maxWords,
      points: nodes.map((n) => ({
        ...n,
        cx: VIEW / 2 + (n.x - centerX) * scale,
        cy: VIEW / 2 - (n.y - centerY) * scale,
      })),
    }
  }, [nodes, spread])

  // Radius scale and the min/max layout above both stay keyed to the FULL
  // node set regardless of hiddenBuckets, so toggling a category off never
  // resizes or re-fits the points that remain -- only visiblePoints (below)
  // changes what actually gets drawn.
  const visiblePoints = useMemo(
    () => points.filter((p) => !hiddenBuckets.has(p.dominant_bucket)),
    [points, hiddenBuckets],
  )

  function radiusFor(wordCount) {
    const t = Math.sqrt(wordCount / maxWordCount) // area-proportional, not radius-proportional
    return RADIUS_MIN + t * (RADIUS_MAX - RADIUS_MIN)
  }

  function toggleBucket(bucket) {
    setHiddenBuckets((prev) => {
      const next = new Set(prev)
      if (next.has(bucket)) next.delete(bucket)
      else next.add(bucket)
      return next
    })
  }

  function handleClick(p) {
    if (entity === 'book') navigate(`/app/authors/${encodeURIComponent(p.subtitle || '')}/${p.id}`)
    else navigate(`/app/authors/${encodeURIComponent(p.id)}`)
  }

  return (
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
          <input
            type="range"
            min={SPREAD_MIN}
            max={SPREAD_MAX}
            step={0.1}
            value={spread}
            onChange={(e) => setSpread(Number(e.target.value))}
          />
        </label>
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>

      {error && <div className="graph-error">{error}</div>}
      {nodes === null && !error && <div className="graph-loading">Loading…</div>}
      {nodes !== null && nodes.length === 0 && !error && (
        <div className="graph-empty">
          No {entity === 'book' ? 'works' : 'authors'} meet the minimum word count yet.
        </div>
      )}
      {nodes !== null && nodes.length > 0 && visiblePoints.length === 0 && (
        <div className="graph-empty">Every category is hidden -- click a legend color to bring it back.</div>
      )}

      {visiblePoints.length > 0 && (
        <svg
          viewBox={`0 0 ${VIEW} ${VIEW}`}
          className="domain-map-svg"
          role="img"
          aria-label="Discipline-category relationship map"
        >
          {visiblePoints.map((p) => {
            const isActive = p.id === hovered
            return (
              <g
                key={p.id}
                className="domain-map-point"
                onClick={() => handleClick(p)}
                onMouseEnter={() => setHovered(p.id)}
                onMouseLeave={() => setHovered((h) => (h === p.id ? null : h))}
              >
                <circle
                  cx={p.cx}
                  cy={p.cy}
                  r={radiusFor(p.word_count)}
                  fill={colorForBucket(p.dominant_bucket)}
                  opacity={hovered === null || isActive ? 1 : 0.35}
                />
                {isActive && (
                  <text x={p.cx} y={p.cy - radiusFor(p.word_count) - 4} textAnchor="middle" className="domain-map-label">
                    {p.label}
                    {p.subtitle ? ` · ${p.subtitle}` : ''} — {p.dominant_name} ({Math.round(p.dominant_fraction * 100)}%)
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      )}

      {legend.length > 0 && (
        <div className="domain-map-legend">
          {legend.map((e) => {
            const isHidden = hiddenBuckets.has(e.bucket)
            return (
              <button
                type="button"
                key={e.bucket}
                className={isHidden ? 'domain-map-legend-item off' : 'domain-map-legend-item'}
                onClick={() => toggleBucket(e.bucket)}
                title={isHidden ? `Show ${e.name}` : `Hide ${e.name}`}
              >
                <span className="domain-map-legend-swatch" style={{ background: colorForBucket(e.bucket) }} />
                {e.name}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default DomainMap
