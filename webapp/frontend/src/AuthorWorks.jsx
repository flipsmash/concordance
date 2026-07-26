import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { difficultySummary } from './bookDifficulty'
import SharedWordsPanel from './SharedWordsPanel'
import { usePagedTable } from './usePagedTable'
import './Authors.css'
import './WorkDetail.css'

const API_BASE = ''
const PAGE_SIZE = 30

// Level 2: one author's works, each with entry count + mean/stddev
// difficulty. Sparse difficulty coverage is a known, real state of the
// corpus -- shown honestly (see difficultySummary) rather than papered over.
function AuthorWorks() {
  const { author } = useParams()
  const navigate = useNavigate()
  const [related, setRelated] = useState(null) // null = not loaded yet, [] = loaded, none found
  const [compareAuthor, setCompareAuthor] = useState(null) // the related author currently being compared, or null

  const { items, total, page, setPage, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/browse/books',
    pageSize: PAGE_SIZE,
    defaultSort: 'title',
    defaultDir: 'asc',
    extraParams: { author },
  })

  function surpriseMe() {
    fetch(`${API_BASE}/api/browse/authors?random=true`)
      .then((res) => res.json())
      .then((data) => {
        const next = data.items[0]
        if (next) navigate(`/app/authors/${encodeURIComponent(next.author)}`)
      })
      .catch(() => {})
  }

  useEffect(() => {
    setRelated(null)
    fetch(`${API_BASE}/api/browse/authors/${encodeURIComponent(author)}/related?top_k=6`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data) {
          setRelated([])
          return
        }
        // shared_word_count lives on the edge, not the node -- join by id.
        const sharedById = new Map(data.edges.map((e) => [e.target, e.shared_word_count]))
        setRelated(
          data.nodes
            .filter((n) => n.ring === 1)
            .map((n) => ({ ...n, shared_word_count: sharedById.get(n.id) ?? 0 })),
        )
      })
      .catch(() => setRelated([]))
  }, [author])

  return (
    <div className="authors-page">
      <header className="authors-header">
        <h1>{author}</h1>
        <button type="button" className="authors-surprise" onClick={surpriseMe}>
          🎲 Surprise me
        </button>
      </header>

      <section>
        <h2 className="work-detail-heading">Related authors</h2>
        {related === null ? (
          <p className="muted">Loading…</p>
        ) : related.length > 0 ? (
          <>
            <ul className="related-list">
              {related.map((a) => (
                <li key={a.id} className="related-row" onClick={() => navigate(`/app/authors/${encodeURIComponent(a.id)}`)}>
                  <span className="related-name">{a.id}</span>
                  <span className="related-meta">
                    <span className="related-shared">{a.shared_word_count} shared words</span>
                    <button
                      type="button"
                      className="related-compare"
                      onClick={(e) => {
                        e.stopPropagation()
                        setCompareAuthor(a)
                      }}
                    >
                      Compare
                    </button>
                  </span>
                </li>
              ))}
            </ul>
            <Link to={`/app/authors/${encodeURIComponent(author)}/relatedness`} className="related-see-graph">
              See full relatedness graph →
            </Link>
          </>
        ) : (
          <p className="muted">Not enough shared vocabulary with other authors yet.</p>
        )}
      </section>

      {compareAuthor && (
        <SharedWordsPanel
          fetchUrl={`/api/browse/authors/${encodeURIComponent(author)}/shared-words/${encodeURIComponent(compareAuthor.id)}`}
          titleA={author}
          titleB={compareAuthor.id}
          onClose={() => setCompareAuthor(null)}
        />
      )}

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
