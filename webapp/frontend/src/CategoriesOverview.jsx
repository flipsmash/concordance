import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import CategoryOverlapGraph from './CategoryOverlapGraph'
import { colorForBucket } from './domainColors'
import './Browse.css'
import './Categories.css'

const API_BASE = ''

// Level 1 of the Categories drilldown: the 6 colored discipline-category
// buckets (usas_domains.py's presentation-layer compression of the 21 USAS
// top-level fields), each a tile linking into its own detail page. Reuses
// /api/browse/domains (unfiltered) rather than a new endpoint -- that's
// already the exact {bucket, name, word_count} shape this page needs.
function CategoriesOverview() {
  const [buckets, setBuckets] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/domains`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then(setBuckets)
      .catch((err) => setError(err.message || 'failed to load categories'))
  }, [])

  return (
    <div className="browse-page">
      <header className="browse-header">
        <h1>Browse by category</h1>
      </header>
      <p className="viz-description">
        The corpus's vocabulary, tagged across six broad discipline categories -- pick one to see
        its finer fields, and the authors, books, and words that lean on it most.
      </p>

      {error && <div className="error-banner">{error}</div>}
      {!buckets && !error && <div className="page-loading">Loading…</div>}

      <div className="category-tile-grid">
        {buckets && buckets.map((b) => (
          <Link
            key={b.bucket}
            to={`/app/categories/${b.bucket}`}
            className="category-tile"
            style={{ borderColor: colorForBucket(b.bucket) }}
          >
            <span className="category-tile-swatch" style={{ background: colorForBucket(b.bucket) }} />
            <span className="category-tile-name">{b.name}</span>
            <span className="category-tile-count">{b.word_count.toLocaleString()} words</span>
          </Link>
        ))}
      </div>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">How much do these categories overlap?</h2>
        <CategoryOverlapGraph basePath="/app/categories" />
      </section>
    </div>
  )
}

export default CategoriesOverview
