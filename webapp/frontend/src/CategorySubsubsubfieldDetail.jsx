import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import CategoryLeaderList from './CategoryLeaderList'
import { colorForBucket } from './domainColors'
import './Browse.css'
import './Categories.css'

const API_BASE = ''

// Level 6 (final) of the Categories drilldown: the tagset's actual deepest
// tier (e.g. "S1.1.1 General", under "S1.1" -- only 17 such codes exist in
// total) -- narrower leaders/word-list actions than its parent sub-sub-field.
// Genuinely terminal: confirmed the real USAS tagset has nothing below this
// level, unlike every shallower tier above it in this drilldown, each of
// which turned out to need its own children-fetch and empty-state once
// checked against the actual data.
function CategorySubsubsubfieldDetail() {
  const { bucket, code, subcode, subsubcode, subsubsubcode } = useParams()
  const [name, setName] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    setName('')
    setError('')
    fetch(`${API_BASE}/api/browse/category-counts?parent=${subsubcode}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => {
        const match = data.find((f) => f.code === subsubsubcode)
        setName(match ? match.name : subsubsubcode)
      })
      .catch((err) => setError(err.message || 'failed to load category'))
  }, [subsubcode, subsubsubcode])

  const color = colorForBucket(bucket)

  return (
    <div className="browse-page">
      <header className="browse-header category-detail-header" style={{ borderColor: color }}>
        <div>
          <Link to={`/app/categories/${bucket}/${code}/${subcode}/${subsubcode}`} className="category-breadcrumb">
            ← Back to sub-sub-field
          </Link>
          <h1>
            <span className="category-detail-swatch" style={{ background: color }} />
            {name || subsubsubcode}
          </h1>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <Link to={`/app?top_code=${subsubsubcode}`} className="browse-quiz-link">
        Browse the words in this category →
      </Link>

      <CategoryLeaderList code={subsubsubcode} />
    </div>
  )
}

export default CategorySubsubsubfieldDetail
