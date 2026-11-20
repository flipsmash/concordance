import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import CategoryOverlapGraph from './CategoryOverlapGraph'
import { CATEGORY_TREE_BLURB, CategoryTreeView } from './CategoryTree'
import { CategoryGenreFilter, useCategoryGenre, withGenre } from './categoryGenre'
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
  const [genre, setGenre] = useCategoryGenre()
  const [buckets, setBuckets] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setBuckets(null)
    setError('')
    fetch(withGenre(`${API_BASE}/api/browse/domains`, genre))
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then(setBuckets)
      .catch((err) => setError(err.message || 'failed to load categories'))
  }, [genre])

  return (
    <div className="browse-page">
      <header className="browse-header">
        <h1>Browse by category</h1>
      </header>
      <p className="viz-description">
        The corpus's vocabulary, tagged across six broad discipline categories -- pick one to see
        its finer fields, and the authors, books, and words that lean on it most.
      </p>
      <CategoryGenreFilter genre={genre} onChange={setGenre} />

      {error && <div className="error-banner">{error}</div>}
      {!buckets && !error && <div className="page-loading">Loading…</div>}

      <div className="category-tile-grid">
        {buckets && buckets.map((b) => (
          <Link
            key={b.bucket}
            to={withGenre(`/app/categories/${b.bucket}`, genre)}
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
        <CategoryOverlapGraph basePath="/app/categories" genre={genre} />
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Categories by meaning</h2>
        <p className="viz-description">
          {CATEGORY_TREE_BLURB}
          {genre && ' (Always the whole collection, not filtered by genre.)'}
        </p>
        <CategoryTreeView />
      </section>
    </div>
  )
}

export default CategoriesOverview
