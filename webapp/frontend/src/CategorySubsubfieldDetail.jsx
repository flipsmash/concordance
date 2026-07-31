import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import CategoryLeaderList from './CategoryLeaderList'
import CategoryOverlapGraph from './CategoryOverlapGraph'
import { colorForBucket } from './domainColors'
import './Browse.css'
import './Categories.css'

const API_BASE = ''

// Level 5 of the Categories drilldown: one USAS sub-sub-field (e.g. "I2.2
// Business: Selling", under "I2 Business") -- narrower leaders/word-list
// actions than its parent sub-field, plus a subgrid of the tagset's actual
// deepest tier. Empty here is the overwhelmingly common case: only 5 of the
// 100 real sub-sub-fields (A1, A1.5, S1.1, S1.2, T1.1) have any children at
// all, so this mirrors CategorySubfieldDetail's own "no further
// subdivisions" handling rather than assuming this tier is terminal.
function CategorySubsubfieldDetail() {
  const { bucket, code, subcode, subsubcode } = useParams()
  const [subsubfieldName, setSubsubfieldName] = useState('')
  const [children, setChildren] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setSubsubfieldName('')
    setError('')
    fetch(`${API_BASE}/api/browse/category-counts?parent=${subcode}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => {
        const match = data.find((f) => f.code === subsubcode)
        setSubsubfieldName(match ? match.name : subsubcode)
      })
      .catch((err) => setError(err.message || 'failed to load sub-sub-field'))
  }, [subcode, subsubcode])

  useEffect(() => {
    setChildren(null)
    fetch(`${API_BASE}/api/browse/category-counts?parent=${subsubcode}`)
      .then((res) => res.json())
      .then(setChildren)
      .catch(() => setChildren([]))
  }, [subsubcode])

  const color = colorForBucket(bucket)

  return (
    <div className="browse-page">
      <header className="browse-header category-detail-header" style={{ borderColor: color }}>
        <div>
          <Link to={`/app/categories/${bucket}/${code}/${subcode}`} className="category-breadcrumb">
            ← Back to sub-field
          </Link>
          <h1>
            <span className="category-detail-swatch" style={{ background: color }} />
            {subsubfieldName || subsubcode}
          </h1>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <Link to={`/app?top_code=${subsubcode}`} className="browse-quiz-link">
        Browse the words in this sub-sub-field →
      </Link>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Sub-divisions within {subsubfieldName || subsubcode}</h2>
        {!children && <div className="page-loading">Loading…</div>}
        {children && children.length === 0 && (
          <p className="category-empty">No further subdivisions.</p>
        )}
        {children && children.length > 0 && (
          <div className="category-tile-grid">
            {children.map((f) => (
              <Link
                key={f.code}
                to={`/app/categories/${bucket}/${code}/${subcode}/${subsubcode}/${f.code}`}
                className="category-tile category-tile-sub"
                style={{ borderColor: color }}
              >
                <span className="category-tile-name">{f.name}</span>
                <span className="category-tile-count">{f.word_count.toLocaleString()} words</span>
              </Link>
            ))}
          </div>
        )}
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">How much do these sub-divisions overlap?</h2>
        <CategoryOverlapGraph parent={subsubcode} basePath={`/app/categories/${bucket}/${code}/${subcode}/${subsubcode}`} />
      </section>

      <CategoryLeaderList code={subsubcode} />
    </div>
  )
}

export default CategorySubsubfieldDetail
