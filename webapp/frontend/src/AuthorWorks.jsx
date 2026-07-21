import { Link, useNavigate, useParams } from 'react-router-dom'
import { usePagedTable } from './usePagedTable'
import './Authors.css'

const PAGE_SIZE = 30

function difficultySummary(book) {
  if (book.scored_word_count === 0) {
    return { stat: 'Not yet scored', qualifier: null }
  }
  const mean = book.mean_difficulty.toFixed(1)
  const stat =
    book.stddev_difficulty === null
      ? `${mean} difficulty (± N/A — not enough scored words)`
      : `${mean} ± ${book.stddev_difficulty.toFixed(1)} difficulty`
  return { stat, qualifier: `based on ${book.scored_word_count} of ${book.word_count} words` }
}

// Level 2: one author's works, each with entry count + mean/stddev
// difficulty. Sparse difficulty coverage is a known, real state of the
// corpus -- shown honestly (see difficultySummary) rather than papered over.
function AuthorWorks() {
  const { author } = useParams()
  const navigate = useNavigate()

  const { items, total, page, setPage, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/browse/books',
    pageSize: PAGE_SIZE,
    defaultSort: 'title',
    defaultDir: 'asc',
    extraParams: { author },
  })

  return (
    <div className="authors-page">
      <header className="authors-header">
        <h1>{author}</h1>
        <Link to="/app/authors" className="authors-back-link">← All authors</Link>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <ul className="authors-list">
        {items.map((b) => {
          const { stat, qualifier } = difficultySummary(b)
          return (
            <li key={b.id} className="authors-row work-row" onClick={() => navigate(`/app/authors/${encodeURIComponent(author)}/${b.id}`)}>
              <span className="work-title">{b.title}</span>
              <span className="work-stats">
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
        {!loading && items.length === 0 && <li className="authors-empty">No works found for this author.</li>}
      </ul>

      <footer className="authors-footer">
        <button type="button" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
          ← Prev
        </button>
        <span>
          Page {page} of {totalPages} ({total} works)
        </span>
        <button type="button" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
          Next →
        </button>
      </footer>
    </div>
  )
}

export default AuthorWorks
