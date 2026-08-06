import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import AlphabetStrip from './AlphabetStrip'
import { densityLabel, difficultySummary, overallDifficultyLabel } from './bookDifficulty'
import SortControl from './SortControl'
import { usePagedTable } from './usePagedTable'
import './Authors.css'
import './Browse.css'

const API_BASE = ''
const PAGE_SIZE = 30

const SORT_FIELDS = [
  { key: 'title', label: 'Title (A–Z)' },
  { key: 'word_count', label: 'Word count' },
  { key: 'difficulty', label: 'Avg. difficulty' },
  { key: 'density', label: 'Vocabulary density' },
  { key: 'overall_difficulty', label: 'Overall difficulty' },
  { key: 'fame', label: 'Fame' },
]

// Level 1 of the book drilldown: every book, browsable/searchable by title --
// the flat counterpart to Authors (which groups the same books by writer).
// Same list chrome as Authors.jsx (shared Authors.css classes) since it's
// the identical browsing mechanism, just a different grouping.
function Books() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [letter, setLetter] = useState(null)
  const [fameMin, setFameMin] = useState('')
  const [fameMax, setFameMax] = useState('')
  // Deep-linked from the Visualizations page's "words unique to each book"
  // histogram (?unique_word_bucket=<label>) -- read once at mount, not kept
  // in sync with the URL afterward, same one-way-in pattern as every other
  // filter here.
  const uniqueWordBucket = searchParams.get('unique_word_bucket')

  useEffect(() => {
    const handle = setTimeout(() => setDebounced(query.trim()), 200)
    return () => clearTimeout(handle)
  }, [query])

  const { items, total, page, setPage, sort, dir, handleSort, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/browse/books',
    pageSize: PAGE_SIZE,
    defaultSort: 'title',
    defaultDir: 'asc',
    extraParams: { q: debounced, letter, fame_min: fameMin, fame_max: fameMax, unique_word_bucket: uniqueWordBucket },
  })

  function changeLetter(l) {
    setLetter(l)
    setPage(1)
  }

  function clearUniqueWordBucket() {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev)
      next.delete('unique_word_bucket')
      return next
    })
    setPage(1)
  }

  function changeFameMin(v) {
    setFameMin(v)
    setPage(1)
  }

  function changeFameMax(v) {
    setFameMax(v)
    setPage(1)
  }

  function surpriseMe() {
    fetch(`${API_BASE}/api/browse/books?random=true`)
      .then((res) => res.json())
      .then((data) => {
        const next = data.items[0]
        if (next) navigate(`/app/authors/${encodeURIComponent(next.author || '')}/${next.id}`)
      })
      .catch(() => {})
  }

  return (
    <div className="authors-page">
      <header className="authors-header">
        <h1>Browse by book</h1>
      </header>

      <button type="button" className="authors-surprise" onClick={surpriseMe}>
        🎲 Surprise me
      </button>

      <input
        type="text"
        className="authors-search"
        placeholder="Search books…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        autoFocus
      />

      <div className="authors-toolbar">
        <AlphabetStrip letter={letter} onChange={changeLetter} />
        <label className="authors-fame-filter">
          Fame:
          <input
            type="number" min="1" max="10" placeholder="1"
            value={fameMin}
            onChange={(e) => changeFameMin(e.target.value)}
          />
          <span>–</span>
          <input
            type="number" min="1" max="10" placeholder="10"
            value={fameMax}
            onChange={(e) => changeFameMax(e.target.value)}
          />
        </label>
        <SortControl fields={SORT_FIELDS} sort={sort} dir={dir} onSort={handleSort} />
      </div>

      {uniqueWordBucket && (
        <div className="browse-shelf">
          <button type="button" className="browse-chip" onClick={clearUniqueWordBucket}>
            Unique words: {uniqueWordBucket} ×
          </button>
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      <ul className="authors-list">
        {items.map((b) => {
          const { stat, qualifier } = difficultySummary(b)
          const density = densityLabel(b.density)
          const overall = overallDifficultyLabel(b.overall_difficulty)
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
                {density && <span className="work-density">{density}</span>}
                {overall && <span className="work-overall">{overall}</span>}
                {b.fame_score != null && (
                  <span className="authors-fame-tag" title={b.fame_reasoning || ''}>
                    ⭐ {b.fame_score.toFixed(1)}
                  </span>
                )}
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
