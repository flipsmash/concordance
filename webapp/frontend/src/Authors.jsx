import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { usePagedTable } from './usePagedTable'
import './Authors.css'

const API_BASE = ''
const PAGE_SIZE = 30

// Level 1 of the author drilldown: every author, browsable/searchable.
// Companion to the faceted Browse page, not a replacement -- a hierarchical
// path (author -> work -> words) rather than a flat filter bag.
function Authors() {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')

  useEffect(() => {
    const handle = setTimeout(() => setDebounced(query.trim()), 200)
    return () => clearTimeout(handle)
  }, [query])

  const { items, total, page, setPage, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/browse/authors',
    pageSize: PAGE_SIZE,
    defaultSort: 'author',
    defaultDir: 'asc',
    extraParams: { q: debounced },
  })

  return (
    <div className="authors-page">
      <header className="authors-header">
        <h1>Browse by author</h1>
        <Link to="/app" className="authors-back-link">← Back to browse</Link>
      </header>

      <input
        type="text"
        className="authors-search"
        placeholder="Search authors…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        autoFocus
      />

      {error && <div className="error-banner">{error}</div>}

      <ul className="authors-list">
        {items.map((a) => (
          <li
            key={a.author}
            className="authors-row"
            onClick={() => navigate(`/app/authors/${encodeURIComponent(a.author)}`)}
          >
            <span className="authors-name">{a.author}</span>
            <span className="authors-counts">
              {a.book_count} {a.book_count === 1 ? 'work' : 'works'} · {a.word_count}{' '}
              {a.word_count === 1 ? 'word' : 'words'}
            </span>
          </li>
        ))}
        {!loading && items.length === 0 && <li className="authors-empty">No authors match.</li>}
      </ul>

      <footer className="authors-footer">
        <button type="button" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
          ← Prev
        </button>
        <span>
          Page {page} of {totalPages} ({total} authors)
        </span>
        <button type="button" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
          Next →
        </button>
      </footer>
    </div>
  )
}

export default Authors
