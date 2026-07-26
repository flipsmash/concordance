import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { difficultySummary } from './bookDifficulty'
import { usePagedTable } from './usePagedTable'
import './Authors.css'

const PAGE_SIZE = 30

// Level 1 of the book drilldown: every book, browsable/searchable by title --
// the flat counterpart to Authors (which groups the same books by writer).
// Same list chrome as Authors.jsx (shared Authors.css classes) since it's
// the identical browsing mechanism, just a different grouping.
function Books() {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')

  useEffect(() => {
    const handle = setTimeout(() => setDebounced(query.trim()), 200)
    return () => clearTimeout(handle)
  }, [query])

  const { items, total, page, setPage, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/browse/books',
    pageSize: PAGE_SIZE,
    defaultSort: 'title',
    defaultDir: 'asc',
    extraParams: { q: debounced },
  })

  return (
    <div className="authors-page">
      <header className="authors-header">
        <h1>Browse by book</h1>
      </header>

      <input
        type="text"
        className="authors-search"
        placeholder="Search books…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        autoFocus
      />

      {error && <div className="error-banner">{error}</div>}

      <ul className="authors-list">
        {items.map((b) => {
          const { stat, qualifier } = difficultySummary(b)
          return (
            <li
              key={b.id}
              className="authors-row work-row"
              onClick={() => navigate(`/app/authors/${encodeURIComponent(b.author || '')}/${b.id}`)}
            >
              <span className="work-title">{b.title}</span>
              <span className="work-stats">
                {b.author && <span>{b.author}</span>}
                <span className="work-count">
                  {b.word_count} {b.word_count === 1 ? 'entry' : 'entries'}
                </span>
                <span className="work-difficulty">
                  {stat}
                  {qualifier && <span className="work-qualifier"> ({qualifier})</span>}
                </span>
              </span>
            </li>
          )
        })}
        {!loading && items.length === 0 && <li className="authors-empty">No books match.</li>}
      </ul>

      <footer className="authors-footer">
        <button type="button" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
          ← Prev
        </button>
        <span>
          Page {page} of {totalPages} ({total} books)
        </span>
        <button type="button" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
          Next →
        </button>
      </footer>
    </div>
  )
}

export default Books
