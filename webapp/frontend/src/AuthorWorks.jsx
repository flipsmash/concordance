import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import AddToSetMenu from './AddToSetMenu'
import { difficultySummary } from './bookDifficulty'
import DifficultyHistogram from './DifficultyHistogram'
import DomainDistribution from './DomainDistribution'
import Pagination from './Pagination'
import SharedWordsPanel from './SharedWordsPanel'
import { buildQueryParams, usePagedTable } from './usePagedTable'
import { useWordFilters } from './useWordFilters'
import './Authors.css'
import './Browse.css'
import './WorkDetail.css'

const API_BASE = ''
const PAGE_SIZE = 30

// Level 2: one author's works, each with entry count + mean/stddev
// difficulty. Sparse difficulty coverage is a known, real state of the
// corpus -- shown honestly (see difficultySummary) rather than papered over.
//
// Also shows the author's own domain/difficulty makeup and a click-to-filter
// word list, mirroring WorkDetail's (level 3) chart+list treatment but
// scoped by `author` instead of `book_id` -- a separate, second
// usePagedTable instance from the works list below, since the two lists
// have nothing to do with each other (one's books, one's words).
function AuthorWorks() {
  const { author } = useParams()
  const navigate = useNavigate()
  const [related, setRelated] = useState(null) // null = not loaded yet, [] = loaded, none found
  const [compareAuthor, setCompareAuthor] = useState(null) // the related author currently being compared, or null
  const [authorRow, setAuthorRow] = useState(null) // this author's own aggregate row (book_count, fame, ...)
  const [domainSummary, setDomainSummary] = useState(null)
  const [bands, setBands] = useState([])
  const [exclusiveCount, setExclusiveCount] = useState(null) // words appearing nowhere else in the corpus

  const {
    selectedDomain, selectedBand, exclusiveOnly,
    toggleDomain, toggleBand, toggleExclusive, clear,
    wordParams, domainSummaryParams, bandsParams,
  } = useWordFilters()

  const { items, total, page, setPage, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/browse/books',
    pageSize: PAGE_SIZE,
    defaultSort: 'title',
    defaultDir: 'asc',
    extraParams: { author },
  })

  const {
    items: wordItems,
    total: wordTotal,
    page: wordPage,
    setPage: setWordPage,
    loading: wordsLoading,
    error: wordsError,
    totalPages: wordTotalPages,
  } = usePagedTable({
    endpoint: '/api/browse/words',
    pageSize: PAGE_SIZE,
    defaultSort: 'lemma',
    defaultDir: 'asc',
    extraParams: { author, ...wordParams },
  })

  useEffect(() => {
    const params = buildQueryParams({ author }, domainSummaryParams)
    fetch(`${API_BASE}/api/browse/domain-summary?${params}`)
      .then((res) => res.json())
      .then(setDomainSummary)
      .catch(() => {})
  }, [author, JSON.stringify(domainSummaryParams)]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const params = buildQueryParams({ author }, bandsParams)
    fetch(`${API_BASE}/api/browse/difficulty-bands?${params}`)
      .then((res) => res.json())
      .then(setBands)
      .catch(() => {})
  }, [author, JSON.stringify(bandsParams)]) // eslint-disable-line react-hooks/exhaustive-deps

  // Deliberately author + exclusive only -- never wordParams -- so this
  // answers "how many words are unique to this author, period," not
  // "...among the currently-selected domain/difficulty." Same
  // /api/browse/words code path the list below uses, so the two numbers
  // can never diverge.
  useEffect(() => {
    fetch(`${API_BASE}/api/browse/words?author=${encodeURIComponent(author)}&exclusive=true&page_size=1`)
      .then((res) => res.json())
      .then((data) => setExclusiveCount(data.total))
      .catch(() => {})
  }, [author])

  function selectDomain(bucket) {
    toggleDomain(bucket)
    setWordPage(1)
  }

  function selectBand(band) {
    toggleBand(band)
    setWordPage(1)
  }

  function clearFilters() {
    clear()
    setWordPage(1)
  }

  const selectedDomainName = domainSummary?.buckets.find((b) => b.bucket === selectedDomain)?.name
  const selectedBandLabel = bands.find((b) =>
    selectedBand === 'unscored' ? b.band_min === null : selectedBand && String(b.band_min) === selectedBand.min,
  )?.label

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/authors?author=${encodeURIComponent(author)}`)
      .then((res) => res.json())
      .then((data) => setAuthorRow(data.items[0] || null))
      .catch(() => setAuthorRow(null))
  }, [author])

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

      {authorRow?.fame_score != null && (
        <section className="work-detail-section">
          <h2 className="work-detail-heading">Fame &amp; importance</h2>
          <p>
            <strong>{authorRow.fame_score.toFixed(1)} / 10</strong>
            {authorRow.fame_reasoning && <span className="muted"> — {authorRow.fame_reasoning}</span>}
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

      <Pagination page={page} totalPages={totalPages} total={total} itemLabel="works" onPageChange={setPage} />

      {(selectedDomainName || selectedBandLabel || exclusiveOnly) && (
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
          {exclusiveOnly && (
            <button type="button" className="browse-chip" onClick={toggleExclusive}>
              Unique to this author ×
            </button>
          )}
          <button type="button" className="browse-clear-all" onClick={clearFilters}>
            Clear filters
          </button>
        </div>
      )}

      {wordsError && <div className="error-banner">{wordsError}</div>}

      {exclusiveCount !== null && (
        <button
          type="button"
          className={`work-detail-exclusive-tile${exclusiveOnly ? ' active' : ''}`}
          onClick={() => {
            toggleExclusive()
            setWordPage(1)
          }}
        >
          <span className="work-detail-exclusive-value">{exclusiveCount}</span>
          <span className="work-detail-exclusive-label">words unique to this author</span>
        </button>
      )}

      <h2 className="work-detail-heading">Words ({wordTotal})</h2>
      <ul className="browse-results">
        {wordItems.map((w) => (
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
            <span className="browse-result-add-to-set" onClick={(e) => e.stopPropagation()}>
              <AddToSetMenu wordIds={[w.id]} label="+" title="Add to set" />
            </span>
          </li>
        ))}
        {!wordsLoading && wordItems.length === 0 && <li className="browse-empty">No words found for this author.</li>}
      </ul>

      <Pagination
        page={wordPage}
        totalPages={wordTotalPages}
        total={wordTotal}
        itemLabel="words"
        onPageChange={setWordPage}
      />
    </div>
  )
}

export default AuthorWorks
