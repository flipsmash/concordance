import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import CategoryLeaderList from './CategoryLeaderList'
import { colorForBucket } from './domainColors'
import './Browse.css'
import './Categories.css'

const API_BASE = ''

// Level 2 of the Categories drilldown: one bucket's member USAS top-level
// fields (e.g. Nature & Science -> Food & Farming, Life & Living Things,
// Science & Technology...), plus the bucket-wide leaders/word-list actions.
function CategoryBucketDetail() {
  const { bucket } = useParams()
  const [bucketName, setBucketName] = useState('')
  const [fields, setFields] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/domains`)
      .then((res) => res.json())
      .then((data) => {
        const match = data.find((b) => b.bucket === bucket)
        setBucketName(match ? match.name : bucket)
      })
      .catch(() => setBucketName(bucket))
  }, [bucket])

  useEffect(() => {
    setFields(null)
    setError('')
    fetch(`${API_BASE}/api/browse/category-counts?bucket=${bucket}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then(setFields)
      .catch((err) => setError(err.message || 'failed to load fields'))
  }, [bucket])

  const color = colorForBucket(bucket)

  return (
    <div className="browse-page">
      <header className="browse-header category-detail-header" style={{ borderColor: color }}>
        <div>
          <Link to="/app/categories" className="category-breadcrumb">
            ← All categories
          </Link>
          <h1>
            <span className="category-detail-swatch" style={{ background: color }} />
            {bucketName}
          </h1>
        </div>
      </header>

      <Link to={`/app?domain=${bucket}`} className="browse-quiz-link">
        Browse the words in this category →
      </Link>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Fields within this category</h2>
        {error && <div className="error-banner">{error}</div>}
        {!fields && !error && <div className="page-loading">Loading…</div>}
        <div className="category-tile-grid">
          {fields && fields.map((f) => (
            <Link
              key={f.code}
              to={`/app/categories/${bucket}/${f.code}`}
              className="category-tile category-tile-sub"
              style={{ borderColor: color }}
            >
              <span className="category-tile-name">{f.name}</span>
              <span className="category-tile-count">{f.word_count.toLocaleString()} words</span>
            </Link>
          ))}
        </div>
      </section>

      <CategoryLeaderList bucket={bucket} />
    </div>
  )
}

export default CategoryBucketDetail
