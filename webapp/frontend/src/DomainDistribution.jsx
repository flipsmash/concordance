import { useEffect, useState } from 'react'
import { colorForBucket } from './domainColors'
import './DomainDistribution.css'

const API_BASE = ''

// Independent per-bucket bars, deliberately NOT a pie/donut: a word can carry
// up to 3 USAS categories, so bucket shares can sum to more than 100% of a
// book's words -- a pie's whole visual grammar (slices sum to the whole)
// would misrepresent every multi-tagged book. Each bar stands alone, sized
// to its own share, with the caption below saying so explicitly.
function DomainDistribution({ bookId }) {
  const [summary, setSummary] = useState(null)

  useEffect(() => {
    setSummary(null)
    fetch(`${API_BASE}/api/browse/domain-summary?book_id=${bookId}`)
      .then((res) => res.json())
      .then(setSummary)
      .catch(() => {})
  }, [bookId])

  if (!summary) return null
  if (summary.total_words === 0) {
    return <p className="domain-dist-empty">No words yet for this book.</p>
  }

  return (
    <div className="domain-dist">
      {summary.buckets.map((b) => {
        const pct = (b.word_count / summary.total_words) * 100
        const barWidth = pct === 0 ? 0 : Math.max(pct, 1.5)
        const isUncategorized = b.bucket === 'uncategorized'
        return (
          <div className="domain-dist-row" key={b.bucket} title={`${b.name}: ${b.word_count} of ${summary.total_words} words (${Math.round(pct)}%)`}>
            <span
              className="domain-dist-swatch"
              style={{ background: isUncategorized ? colorForBucket(null) : colorForBucket(b.bucket) }}
            />
            <span className={isUncategorized ? 'domain-dist-label muted' : 'domain-dist-label'}>{b.name}</span>
            <span className="domain-dist-track">
              <span
                className="domain-dist-fill"
                style={{
                  width: `${barWidth}%`,
                  background: isUncategorized ? colorForBucket(null) : colorForBucket(b.bucket),
                }}
              />
            </span>
            <span className="domain-dist-count">
              {b.word_count} of {summary.total_words} ({Math.round(pct)}%)
            </span>
          </div>
        )
      })}
      <p className="domain-dist-caption">
        % of this book's words touching each domain — categories can overlap, so shares don't add to 100%.
      </p>
    </div>
  )
}

export default DomainDistribution
