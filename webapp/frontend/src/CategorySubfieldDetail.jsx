import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import CategoryLeaderList from './CategoryLeaderList'
import { colorForBucket } from './domainColors'
import './Browse.css'
import './Categories.css'

const API_BASE = ''

// Level 4 of the Categories drilldown: one USAS sub-field (e.g. "I2
// Business", under "I Money & Commerce") -- narrower leaders/word-list
// actions than its parent field, plus a subgrid of its own sub-sub-fields.
// Unlike the field tier above, empty here is common (84 of 115 sub-fields,
// 73%, have no further subdivision at all) -- shown as a plain message
// rather than an empty grid.
function CategorySubfieldDetail() {
  const { bucket, code, subcode } = useParams()
  const [subfieldName, setSubfieldName] = useState('')
  const [subsubfields, setSubsubfields] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setSubfieldName('')
    setError('')
    fetch(`${API_BASE}/api/browse/category-counts?parent=${code}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => {
        const match = data.find((f) => f.code === subcode)
        setSubfieldName(match ? match.name : subcode)
      })
      .catch((err) => setError(err.message || 'failed to load sub-field'))
  }, [code, subcode])

  useEffect(() => {
    setSubsubfields(null)
    fetch(`${API_BASE}/api/browse/category-counts?parent=${subcode}`)
      .then((res) => res.json())
      .then(setSubsubfields)
      .catch(() => setSubsubfields([]))
  }, [subcode])

  const color = colorForBucket(bucket)

  return (
    <div className="browse-page">
      <header className="browse-header category-detail-header" style={{ borderColor: color }}>
        <div>
          <Link to={`/app/categories/${bucket}/${code}`} className="category-breadcrumb">
            ← Back to field
          </Link>
          <h1>
            <span className="category-detail-swatch" style={{ background: color }} />
            {subfieldName || subcode}
          </h1>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <Link to={`/app?top_code=${subcode}`} className="browse-quiz-link">
        Browse the words in this sub-field →
      </Link>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Sub-divisions within {subfieldName || subcode}</h2>
        {!subsubfields && <div className="page-loading">Loading…</div>}
        {subsubfields && subsubfields.length === 0 && (
          <p className="category-empty">No further subdivisions.</p>
        )}
        {subsubfields && subsubfields.length > 0 && (
          <div className="category-tile-grid">
            {subsubfields.map((f) => (
              <Link
                key={f.code}
                to={`/app/categories/${bucket}/${code}/${subcode}/${f.code}`}
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

      <CategoryLeaderList code={subcode} />
    </div>
  )
}

export default CategorySubfieldDetail
