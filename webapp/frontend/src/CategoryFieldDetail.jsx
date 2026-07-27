import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import CategoryLeaderList from './CategoryLeaderList'
import { colorForBucket } from './domainColors'
import './Browse.css'
import './Categories.css'

const API_BASE = ''

// Level 3 (final) of the Categories drilldown: a single USAS top-level
// discourse field (e.g. "S" People) -- narrower leaders/word-list actions
// than its parent bucket. The ~230 dotted subcategories one level below
// this aren't drillable here; this is the request's "secondary level of
// detail," the deepest level in scope.
function CategoryFieldDetail() {
  const { bucket, code } = useParams()
  const [fieldName, setFieldName] = useState('')
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

      <CategoryLeaderList code={code} />
    </div>
  )
}

export default CategoryFieldDetail
