import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import CategoryLeaderList from './CategoryLeaderList'
import { colorForBucket } from './domainColors'
import './Browse.css'
import './Categories.css'

const API_BASE = ''

// Level 3 of the Categories drilldown: a single USAS top-level discourse
// field (e.g. "S" People) -- narrower leaders/word-list actions than its
// parent bucket, plus a subgrid of the field's own USAS sub-fields (e.g.
// "S1 Social actions..."), which the tagset guarantees is never empty here
// (every one of the 21 top-level codes has at least one child).
function CategoryFieldDetail() {
  const { bucket, code } = useParams()
  const [fieldName, setFieldName] = useState('')
  const [subfields, setSubfields] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setFieldName('')
    setError('')
    fetch(`${API_BASE}/api/browse/category-counts?bucket=${bucket}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => {
        const match = data.find((f) => f.code === code)
        setFieldName(match ? match.name : code)
      })
      .catch((err) => setError(err.message || 'failed to load field'))
  }, [bucket, code])

  useEffect(() => {
    setSubfields(null)
    fetch(`${API_BASE}/api/browse/category-counts?parent=${code}`)
      .then((res) => res.json())
      .then(setSubfields)
      .catch(() => setSubfields([]))
  }, [code])

  const color = colorForBucket(bucket)

  return (
    <div className="browse-page">
      <header className="browse-header category-detail-header" style={{ borderColor: color }}>
        <div>
          <Link to={`/app/categories/${bucket}`} className="category-breadcrumb">
            ← Back to category
          </Link>
          <h1>
            <span className="category-detail-swatch" style={{ background: color }} />
            {fieldName || code}
          </h1>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <Link to={`/app?top_code=${code}`} className="browse-quiz-link">
        Browse the words in this field →
      </Link>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Sub-fields within {fieldName || code}</h2>
        {!subfields && <div className="page-loading">Loading…</div>}
        <div className="category-tile-grid">
          {subfields && subfields.map((f) => (
            <Link
              key={f.code}
              to={`/app/categories/${bucket}/${code}/${f.code}`}
              className="category-tile category-tile-sub"
              style={{ borderColor: color }}
            >
              <span className="category-tile-name">{f.name}</span>
              <span className="category-tile-count">{f.word_count.toLocaleString()} words</span>
            </Link>
          ))}
        </div>
      </section>

      <CategoryLeaderList code={code} />
    </div>
  )
}

export default CategoryFieldDetail
