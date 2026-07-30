import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import DifficultyHistogram from './DifficultyHistogram'
import DomainDistribution from './DomainDistribution'
import SharedWordsPanel from './SharedWordsPanel'
import { buildQueryParams, usePagedTable } from './usePagedTable'
import { useWordFilters } from './useWordFilters'
import './Authors.css'
import './Browse.css'
import './WorkDetail.css'

const API_BASE = ''
const PAGE_SIZE = 30

// Level 3 of the author drilldown: one work's domain makeup, difficulty
// spread, and full word list. Reuses the faceted Browse page's result-row
// visual language (imported from Browse.css) scoped to a single book_id.
// The difficulty distribution gets its own real histogram component
// (DifficultyHistogram) rather than reusing Browse's compact filter strip,
// which varies bar WIDTH not height and isn't a histogram.
//
// Both charts are click-to-filter (useWordFilters) and narrow the word list
// below; they also cross-filter each other (selecting a difficulty band
// re-fetches the domain chart's counts and vice versa) via the same
// domainSummaryParams/bandsParams split the hook derives.
function WorkDetail() {
  const { author, bookId } = useParams()
  const navigate = useNavigate()
  const [book, setBook] = useState(null)
  const [domainSummary, setDomainSummary] = useState(null)
  const [bands, setBands] = useState([])
  const [related, setRelated] = useState(null) // null = not loaded yet, [] = loaded, none found
  const [compareBook, setCompareBook] = useState(null) // the related book currently being compared, or null

  const { selectedDomain, selectedBand, toggleDomain, toggleBand, clear, wordParams, domainSummaryParams, bandsParams } =
    useWordFilters()

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/books?book_id=${bookId}`)
      .then((res) => res.json())
      .then((data) => setBook(data.items[0] || null))
      .catch(() => {})
  }, [bookId])

  useEffect(() => {
    const params = buildQueryParams({ book_id: bookId }, domainSummaryParams)
    fetch(`${API_BASE}/api/browse/domain-summary?${params}`)
      .then((res) => res.json())
      .then(setDomainSummary)
      .catch(() => {})
  }, [bookId, JSON.stringify(domainSummaryParams)]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const params = buildQueryParams({ book_id: bookId }, bandsParams)
    fetch(`${API_BASE}/api/browse/difficulty-bands?${params}`)
      .then((res) => res.json())
      .then(setBands)
      .catch(() => {})
  }, [bookId, JSON.stringify(bandsParams)]) // eslint-disable-line react-hooks/exhaustive-deps

  function surpriseMe() {
    fetch(`${API_BASE}/api/browse/books?random=true`)
      .then((res) => res.json())
      .then((data) => {
        const next = data.items[0]
        if (next) navigate(`/app/authors/${encodeURIComponent(next.author || '')}/${next.id}`)
      })
      .catch(() => {})
  }

  useEffect(() => {
    setRelated(null)
    fetch(`${API_BASE}/api/browse/books/${bookId}/related?top_k=6`)
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
  }, [bookId])

  const { items, total, page, setPage, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/browse/words',
    pageSize: PAGE_SIZE,
    defaultSort: 'lemma',
    defaultDir: 'asc',
    extraParams: { book_id: [bookId], ...wordParams },
  })

  function selectDomain(bucket) {
    toggleDomain(bucket)
    setPage(1)
  }

  function selectBand(band) {
    toggleBand(band)
    setPage(1)
  }

  function clearFilters() {
    clear()
    setPage(1)
  }

  const selectedDomainName = domainSummary?.buckets.find((b) => b.bucket === selectedDomain)?.name
  const selectedBandLabel = bands.find((b) =>
    selectedBand === 'unscored' ? b.band_min === null : selectedBand && String(b.band_min) === selectedBand.min,
  )?.label

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
          {book?.archive_path && (
            <a
              href={`/api/browse/books/${bookId}/text`}
              className="work-detail-source-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              View source text
            </a>
          )}
        </div>
        <button type="button" className="authors-surprise" onClick={surpriseMe}>
          🎲 Surprise me
        </button>
      </header>

      {book?.fame_score != null && (
        <section className="browse-facets work-detail-section">
          <h2 className="work-detail-heading">Fame &amp; importance</h2>
          <p>
            <strong>{book.fame_score.toFixed(1)} / 10</strong>
            {book.fame_reasoning && <span className="muted"> — {book.fame_reasoning}</span>}
          </p>
        </section>
      )}

      <section className="browse-facets work-detail-section">
        <h2 className="work-detail-heading">Domains represented</h2>
        <DomainDistribution summary={domainSummary} selected={selectedDomain} onSelect={selectDomain} />
      </section>

      <section className="browse-facets work-detail-section">
        <h2 className="work-detail-heading">Difficulty distribution</h2>
        <DifficultyHistogram bands={bands} selectedBand={selectedBand} onSelect={selectBand} />
      </section>

      <section className="browse-facets work-detail-section">
        <h2 className="work-detail-heading">Related books</h2>
        {related === null ? (
          <p className="muted">Loading…</p>
        ) : related.length > 0 ? (
          <>
            <ul className="related-list">
              {related.map((b) => (
                <li
                  key={b.id}
                  className="related-row"
                  onClick={() => navigate(`/app/authors/${encodeURIComponent(b.author || '')}/${b.id}`)}
                >
                  <span className="related-name">{b.title}</span>
                  <span className="related-meta">
                    {b.author && <span>{b.author}</span>}
                    <span className="related-shared">{b.shared_word_count} shared words</span>
                    <button
                      type="button"
                      className="related-compare"
                      onClick={(e) => {
                        e.stopPropagation()
                        setCompareBook(b)
                      }}
                    >
                      Compare
                    </button>
                  </span>
                </li>
              ))}
            </ul>
            <Link to={`/app/authors/${encodeURIComponent(author)}/${bookId}/relatedness`} className="related-see-graph">
              See full relatedness graph →
            </Link>
          </>
        ) : (
          <p className="muted">Not enough shared vocabulary with other books yet.</p>
        )}
      </section>

      {compareBook && (
        <SharedWordsPanel
          fetchUrl={`/api/browse/books/${bookId}/shared-words/${compareBook.id}`}
          titleA={book?.title || 'This book'}
          titleB={compareBook.title}
          onClose={() => setCompareBook(null)}
        />
      )}

      {error && <div className="error-banner">{error}</div>}

      {(selectedDomainName || selectedBandLabel) && (
        <div className="browse-shelf">
          {selectedDomainName && (
            <button type="button" className="browse-chip" onClick={() => selectDomain(selectedDomain)}>
              {selectedDomainName} ×
            </button>
          )}
          {selectedBandLabel && (
            <button
              type="button"
              className="browse-chip"
              onClick={() =>
                selectBand(
                  selectedBand === 'unscored'
                    ? { band_min: null }
                    : { band_min: Number(selectedBand.min), band_max: Number(selectedBand.max) },
                )
              }
            >
              {selectedBandLabel} ×
            </button>
          )}
          <button type="button" className="browse-clear-all" onClick={clearFilters}>
            Clear filters
          </button>
        </div>
      )}

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
