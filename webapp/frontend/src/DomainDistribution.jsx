import { useEffect, useState } from 'react'
import { colorForBucket } from './domainColors'
import './DomainDistribution.css'

// The whole collection's breakdown, the baseline for the "vs. collection"
// view -- fetched once per page load and shared by every chart on it.
let collectionPromise = null
function useCollectionSummary(enabled) {
  const [summary, setSummary] = useState(null)
  useEffect(() => {
    if (!enabled) return undefined
    let live = true
    collectionPromise ??= fetch('/api/browse/domain-summary')
      .then((res) => (res.ok ? res.json() : null))
      .catch(() => null)
    collectionPromise.then((d) => {
      if (!d) collectionPromise = null // let a later toggle retry
      if (live) setSummary(d)
    })
    return () => { live = false }
  }, [enabled])
  return summary
}

// The index view: how often a domain turns up here relative to the whole
// collection (share here / share overall), drawn left or right of a center
// line at 1x on a log scale, so half as common and twice as common sit the
// same distance from average. Clamped at 1/4x and 4x.
const INDEX_LIMIT = Math.log2(4)
const FEW_WORDS = 50

function formatIndex(ratio) {
  return ratio >= 10 ? `${Math.round(ratio)}×` : `${ratio.toFixed(2)}×`
}

// Independent per-bucket bars, deliberately NOT a pie/donut: a word can carry
// up to 3 USAS categories, so bucket shares can sum to more than 100% of a
// book's words -- a pie's whole visual grammar (slices sum to the whole)
// would misrepresent every multi-tagged book. Each bar stands alone, sized
// to its own share, with the caption below saying so explicitly.
//
// Purely presentational -- the parent owns fetching `summary` (so it can
// scope the request by book_id OR author, and re-fetch when the OTHER
// chart's selection changes for cross-filtering) and the `selected` bucket
// key. Every row, including "Uncategorized", is clickable: the backend's
// `uncategorized` filter (browse.py's _build_word_filters) makes that
// segment a real, narrowable filter rather than a dead end.
//
// Owns its own "Domains represented" heading (rather than leaving it to
// each of the two callers, AuthorWorks.jsx/WorkDetail.jsx) specifically so
// the click-to-sort-by-frequency toggle below lives in one place instead of
// being duplicated across both. `summary.buckets` arrives in the backend's
// own fixed legend order (browse_domain_summary: the 6 named buckets, then
// Uncategorized always last) -- that's "standard order" for the purposes of
// the toggle; sorting is a pure client-side re-order of the already-fetched
// data, no re-fetch involved.
function DomainDistribution({ summary, selected, onSelect }) {
  const [sortByCount, setSortByCount] = useState(false)
  const [mode, setMode] = useState('share') // 'share' | 'index'
  const collection = useCollectionSummary(mode === 'index')

  if (!summary) return null
  if (summary.total_words === 0) {
    return <p className="domain-dist-empty">No words yet for this selection.</p>
  }

  const overall = new Map(
    (collection?.buckets || []).map((b) => [b.bucket, collection.total_words ? b.word_count / collection.total_words : 0]),
  )
  const indexOf = (b) => {
    const base = overall.get(b.bucket)
    return base ? b.word_count / summary.total_words / base : null
  }
  const showIndex = mode === 'index' && collection
  // an empty "Uncategorized" is just noise
  const rows = summary.buckets.filter((b) => !(b.bucket === 'uncategorized' && b.word_count === 0))
  const buckets = sortByCount
    ? [...rows].sort((a, b) => (showIndex ? (indexOf(b) ?? 0) - (indexOf(a) ?? 0) : b.word_count - a.word_count))
    : rows

  return (
    <div className="domain-dist">
      <h2
        className="domain-dist-heading"
        role="button"
        tabIndex={0}
        onClick={() => setSortByCount((s) => !s)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') setSortByCount((s) => !s)
        }}
        title={sortByCount ? 'Click to return to standard order' : 'Click to sort most to least common'}
      >
        Domains represented
        <span className="domain-dist-sort-indicator" aria-hidden="true">
          {sortByCount ? (showIndex ? ' ▾ most over-represented' : ' ▾ by frequency') : ' ⇅'}
        </span>
      </h2>
      <div className="domain-dist-modes" role="group" aria-label="Show">
        {[['share', 'Share of words'], ['index', 'vs. the collection']].map(([id, label]) => (
          <button key={id} type="button" className={mode === id ? 'browse-pill active' : 'browse-pill'}
            onClick={() => setMode(id)}>
            {label}
          </button>
        ))}
      </div>
      {mode === 'index' && !collection && <p className="domain-dist-caption">Loading the collection&rsquo;s averages…</p>}
      {buckets.map((b) => {
        const pct = (b.word_count / summary.total_words) * 100
        const barWidth = pct === 0 ? 0 : Math.max(pct, 1.5)
        const isUncategorized = b.bucket === 'uncategorized'
        const isSelected = selected === b.bucket
        const swatch = isUncategorized ? colorForBucket(null) : colorForBucket(b.bucket)
        const ratio = showIndex ? indexOf(b) : null
        const basePct = (overall.get(b.bucket) || 0) * 100
        const title = showIndex
          ? `${b.name}: ${Math.round(pct)}% of these words vs ${Math.round(basePct)}% across the collection` +
            (ratio != null ? ` (${formatIndex(ratio)} average)` : '')
          : `${b.name}: ${b.word_count} of ${summary.total_words} words (${Math.round(pct)}%)`
        let fill
        if (!showIndex) {
          fill = <span className="domain-dist-fill" style={{ width: `${barWidth}%`, background: swatch }} />
        } else if (ratio != null && ratio > 0) {
          const half = Math.min(Math.abs(Math.log2(ratio)) / INDEX_LIMIT, 1) * 50
          fill = (
            <span className="domain-dist-fill domain-dist-fill-index"
              style={{ width: `${Math.max(half, 0.8)}%`, background: swatch, ...(ratio >= 1 ? { left: '50%' } : { right: '50%' }) }} />
          )
        }
        return (
          <button
            type="button"
            className={isSelected ? 'domain-dist-row active' : 'domain-dist-row'}
            style={isSelected ? { borderColor: swatch } : undefined}
            key={b.bucket}
            title={title}
            onClick={() => onSelect(b.bucket)}
          >
            <span className="domain-dist-swatch" style={{ background: swatch }} />
            <span className={isUncategorized ? 'domain-dist-label muted' : 'domain-dist-label'}>{b.name}</span>
            <span className={showIndex ? 'domain-dist-track domain-dist-track-index' : 'domain-dist-track'}>{fill}</span>
            <span className="domain-dist-count">
              {showIndex ? (ratio == null ? '—' : ratio === 0 ? '0×' : formatIndex(ratio)) : `${Math.round(pct)}%`}
            </span>
          </button>
        )
      })}
      <p className="domain-dist-caption">
        {showIndex ? (
          <>
            How often each domain turns up here compared with the whole collection: the center line is the
            collection average, 2× means twice as common here, 0.5× half as common.
            {summary.total_words < FEW_WORDS && <> With only {summary.total_words} words, read these loosely.</>}
          </>
        ) : (
          <>% of these words touching each domain — categories can overlap, so shares don&rsquo;t add to 100%.</>
        )}{' '}
        Click a domain to filter the word list below.
      </p>
    </div>
  )
}

export default DomainDistribution
