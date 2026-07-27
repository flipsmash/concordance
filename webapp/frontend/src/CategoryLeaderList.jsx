import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './GraphView.css' // .graph-signal-toggle
import './Categories.css'

const API_BASE = ''

const ENTITY_TABS = [
  { id: 'book', label: 'Books' },
  { id: 'author', label: 'Authors' },
]

// Ranked "who/what leans on this category's vocabulary the most" list,
// shared by the bucket-level (CategoryBucketDetail) and field-level
// (CategoryFieldDetail) Categories pages -- pass exactly one of `bucket`/
// `code`. Backed by /api/browse/category-leaders, which ranks by LIFT
// (share of this category / the qualifying population's own mean share),
// not raw word count -- see that endpoint's docstring for why a raw-count
// ranking would put the same prolific authors/long books at the top of
// every single category.
function CategoryLeaderList({ bucket, code }) {
  const navigate = useNavigate()
  const [entity, setEntity] = useState('book')
  const [items, setItems] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setItems(null)
    setError('')
    const params = new URLSearchParams({ entity, page_size: '10' })
    if (bucket) params.set('bucket', bucket)
    if (code) params.set('top_code', code)
    fetch(`${API_BASE}/api/browse/category-leaders?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => setItems(data.items))
      .catch((err) => setError(err.message || 'failed to load leaders'))
  }, [entity, bucket, code])

  function handleClick(item) {
    if (entity === 'book') navigate(`/app/authors/${encodeURIComponent(item.subtitle || '')}/${item.id}`)
    else navigate(`/app/authors/${encodeURIComponent(item.id)}`)
  }

  return (
    <section className="browse-facets viz-section category-leaders">
      <h2 className="viz-heading">Who leans on this vocabulary most</h2>
      <div className="graph-signal-toggle" role="group" aria-label="Entity">
        {ENTITY_TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={entity === t.id ? 'active' : ''}
            onClick={() => setEntity(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}
      {items === null && !error && <div className="page-loading">Loading…</div>}
      {items !== null && items.length === 0 && !error && (
        <p className="category-empty">
          No {entity === 'book' ? 'books' : 'authors'} meet the minimum word count yet.
        </p>
      )}

      {items && items.length > 0 && (
        <ol className="category-leader-list">
          {items.map((item, i) => (
            <li key={item.id} className="category-leader-row" onClick={() => handleClick(item)}>
              <span className="category-leader-rank">{i + 1}</span>
              <span className="category-leader-name">
                {item.label}
                {item.subtitle && <span className="category-leader-subtitle"> · {item.subtitle}</span>}
              </span>
              <span className="category-leader-stats">
                {Math.round(item.share * 100)}% of vocabulary · {item.lift.toFixed(1)}× corpus average
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}

export default CategoryLeaderList
