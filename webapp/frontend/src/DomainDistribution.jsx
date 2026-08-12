import { useState } from 'react'
import { colorForBucket } from './domainColors'
import './DomainDistribution.css'

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

  if (!summary) return null
  if (summary.total_words === 0) {
    return <p className="domain-dist-empty">No words yet for this selection.</p>
  }

  const buckets = sortByCount
    ? [...summary.buckets].sort((a, b) => b.word_count - a.word_count)
    : summary.buckets

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
          {sortByCount ? ' ▾ by frequency' : ' ⇅'}
        </span>
      </h2>
      {buckets.map((b) => {
        const pct = (b.word_count / summary.total_words) * 100
        const barWidth = pct === 0 ? 0 : Math.max(pct, 1.5)
        const isUncategorized = b.bucket === 'uncategorized'
        const isSelected = selected === b.bucket
        const swatch = isUncategorized ? colorForBucket(null) : colorForBucket(b.bucket)
        return (
          <button
            type="button"
            className={isSelected ? 'domain-dist-row active' : 'domain-dist-row'}
            style={isSelected ? { borderColor: swatch } : undefined}
            key={b.bucket}
            title={`${b.name}: ${b.word_count} of ${summary.total_words} words (${Math.round(pct)}%)`}
            onClick={() => onSelect(b.bucket)}
          >
            <span className="domain-dist-swatch" style={{ background: swatch }} />
            <span className={isUncategorized ? 'domain-dist-label muted' : 'domain-dist-label'}>{b.name}</span>
            <span className="domain-dist-track">
              <span className="domain-dist-fill" style={{ width: `${barWidth}%`, background: swatch }} />
            </span>
            <span className="domain-dist-count">{Math.round(pct)}%</span>
          </button>
        )
      })}
      <p className="domain-dist-caption">
        % of these words touching each domain — categories can overlap, so shares don't add to 100%. Click a
        domain to filter the word list below.
      </p>
    </div>
  )
}

export default DomainDistribution
