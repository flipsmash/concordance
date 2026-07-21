import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import DomainDistribution from './DomainDistribution'
import { usePagedTable } from './usePagedTable'
import './Authors.css'
import './Browse.css'
import './WorkDetail.css'

const API_BASE = ''
const PAGE_SIZE = 30

// Level 3 of the author drilldown: one work's domain makeup, difficulty
// spread, and full word list. Reuses the faceted Browse page's difficulty-
// histogram and result-row visual language (imported from Browse.css)
// scoped to a single book_id, rather than reinventing either.
function WorkDetail() {
  const { author, bookId } = useParams()
  const navigate = useNavigate()
  const [book, setBook] = useState(null)
  const [bands, setBands] = useState([])

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/books?book_id=${bookId}`)
      .then((res) => res.json())
      .then((data) => setBook(data.items[0] || null))
      .catch(() => {})
  }, [bookId])

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/difficulty-bands?book_id=${bookId}`)
      .then((res) => res.json())
      .then(setBands)
      .catch(() => {})
  }, [bookId])

  const bandsTotal = bands.reduce((sum, b) => sum + b.word_count, 0) || 1

  const { items, total, page, setPage, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/browse/words',
    pageSize: PAGE_SIZE,
    defaultSort: 'lemma',
    defaultDir: 'asc',
    extraParams: { book_id: [bookId] },
  })

  return (
    <div className="browse-page work-detail-page">
      <header className="authors-header">
        <div>
          <h1>{book ? book.title : 'Loading…'}</h1>
          {book?.author && (
            <Link to={`/app/authors/${encodeURIComponent(book.author)}`} className="work-detail-author-link">
              {book.author}
            </Link>
          )}
        </div>
        <Link to={`/app/authors/${encodeURIComponent(author)}`} className="authors-back-link">
          ← {author}'s works
        </Link>
      </header>

      <section className="browse-facets work-detail-section">
        <h2 className="work-detail-heading">Domains represented</h2>
        <DomainDistribution bookId={bookId} />
      </section>

      <section className="browse-facets work-detail-section">
        <h2 className="work-detail-heading">Difficulty distribution</h2>
        <div className="browse-difficulty-histogram">
          {bands.map((b) => (
            <div
              key={b.label}
              className={b.band_min === null ? 'browse-band unscored' : 'browse-band'}
              style={{ flexGrow: Math.max(b.word_count, 1) / bandsTotal }}
              title={`${b.label}: ${b.word_count} words`}
            />
          ))}
        </div>
      </section>

      {error && <div className="error-banner">{error}</div>}

      <h2 className="work-detail-heading">Words ({total})</h2>
      <ul className="browse-results">
        {items.map((w) => (
          <li key={w.id} className="browse-result-row" onClick={() => navigate(`/app/words/${w.id}`)}>
            <span className="browse-result-lemma">{w.lemma}</span>
            {w.part_of_speech && <span className="browse-result-pos">{w.part_of_speech}</span>}
            {w.definition && (
              <span className="browse-result-def">
                {w.definition.length > 100 ? `${w.definition.slice(0, 100)}…` : w.definition}
              </span>
            )}
            <span className="browse-result-badges">
              {w.difficulty != null && <span className="browse-difficulty-pill">{Math.round(w.difficulty)}</span>}
              {w.archaic && w.archaic !== 'current' && <span className="browse-archaic-tag">{w.archaic}</span>}
            </span>
          </li>
        ))}
        {!loading && items.length === 0 && <li className="browse-empty">No words found for this book.</li>}
      </ul>

      <footer className="browse-footer">
        <button type="button" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
          ← Prev
        </button>
        <span>
          Page {page} of {totalPages} ({total} words)
        </span>
        <button type="button" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
          Next →
        </button>
      </footer>
    </div>
  )
}

export default WorkDetail
