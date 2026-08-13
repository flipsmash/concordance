import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import AlphabetStrip from './AlphabetStrip'
import Pagination from './Pagination'
import SortControl from './SortControl'
import { usePagedTable } from './usePagedTable'
import './Authors.css'
import './Browse.css'

const API_BASE = ''
const PAGE_SIZE = 30

const SORT_FIELDS = [
  { key: 'author', label: 'Name (A–Z)' },
  { key: 'book_count', label: 'Work count' },
  { key: 'word_count', label: 'Word count' },
  { key: 'unique_word_count', label: 'Unique words contributed' },
  { key: 'difficulty', label: 'Avg. difficulty' },
  { key: 'density', label: 'Vocabulary density' },
  { key: 'overall_difficulty', label: 'Overall difficulty' },
  { key: 'fame', label: 'Fame' },
]

// Same shape as bookDifficulty.js's densityLabel/overallDifficultyLabel --
// kept local since AuthorRow is a distinct shape from BookRow (mean of
// per-book densities, not a single book's own), used in only this one place.
function densityLabel(density) {
  if (density == null) return null
  return `${(density * 100).toFixed(1)}% vocabulary density`
}

function overallDifficultyLabel(overall) {
  if (overall == null) return null
  return `Harder than ${overall}% of the corpus`
}

// Level 1 of the author drilldown: every author, browsable/searchable.
// Companion to the faceted Browse page, not a replacement -- a hierarchical
// path (author -> work -> words) rather than a flat filter bag.
function Authors() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [letter, setLetter] = useState(null)
  // fame_min/fame_max/fame_unscored_only seed from the URL so a click on the
  // Visualizations page's fame-distribution bars (?fame_min=9&fame_max=9 or
  // ?fame_unscored_only=true) lands pre-filtered -- same one-way-in pattern
  // as unique_word_bucket below (read once at mount, not kept in sync with
  // the URL afterward).
  const [fameMin, setFameMin] = useState(() => searchParams.get('fame_min') || '')
  const [fameMax, setFameMax] = useState(() => searchParams.get('fame_max') || '')
  const [fameUnscoredOnly, setFameUnscoredOnly] = useState(() => searchParams.get('fame_unscored_only') === 'true')
  // Deep-linked from the Visualizations page's "words unique to each author"
  // histogram (?unique_word_bucket=<label>) -- read once at mount, not kept
  // in sync with the URL afterward, same one-way-in pattern as every other
  // filter here.
  const uniqueWordBucket = searchParams.get('unique_word_bucket')
  // Same one-way-in pattern, deep-linked from the "Author difficulty
  // distribution" histogram (?overall_difficulty_band=<label>).
  const overallDifficultyBand = searchParams.get('overall_difficulty_band')

  useEffect(() => {
    const handle = setTimeout(() => setDebounced(query.trim()), 200)
    return () => clearTimeout(handle)
  }, [query])

  const { items, total, page, setPage, sort, dir, handleSort, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/browse/authors',
    pageSize: PAGE_SIZE,
    defaultSort: 'author',
    defaultDir: 'asc',
    extraParams: {
      q: debounced,
      letter,
      fame_min: fameUnscoredOnly ? '' : fameMin,
      fame_max: fameUnscoredOnly ? '' : fameMax,
      fame_unscored_only: fameUnscoredOnly,
      unique_word_bucket: uniqueWordBucket,
      overall_difficulty_band: overallDifficultyBand,
    },
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

  function clearOverallDifficultyBand() {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev)
      next.delete('overall_difficulty_band')
      return next
    })
    setPage(1)
  }

  function changeFameMin(v) {
    setFameMin(v)
    setFameUnscoredOnly(false)
    setPage(1)
  }

  function changeFameMax(v) {
    setFameMax(v)
    setFameUnscoredOnly(false)
    setPage(1)
  }

  function clearFameUnscoredOnly() {
    setFameUnscoredOnly(false)
    setPage(1)
  }

  function surpriseMe() {
    fetch(`${API_BASE}/api/browse/authors?random=true`)
      .then((res) => res.json())
      .then((data) => {
        const next = data.items[0]
        if (next) navigate(`/app/authors/${encodeURIComponent(next.author)}`)
      })
      .catch(() => {})
  }

  return (
    <div className="authors-page">
      <header className="authors-header">
        <h1>Browse by author</h1>
      </header>

      <button type="button" className="authors-surprise" onClick={surpriseMe}>
        🎲 Surprise me
      </button>

      <input
        type="text"
        className="authors-search"
        placeholder="Search authors…"
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
            disabled={fameUnscoredOnly}
            onChange={(e) => changeFameMin(e.target.value)}
          />
          <span>–</span>
          <input
            type="number" min="1" max="10" placeholder="10"
            value={fameMax}
            disabled={fameUnscoredOnly}
            onChange={(e) => changeFameMax(e.target.value)}
          />
        </label>
        <SortControl fields={SORT_FIELDS} sort={sort} dir={dir} onSort={handleSort} />
      </div>

      {(uniqueWordBucket || fameUnscoredOnly || overallDifficultyBand) && (
        <div className="browse-shelf">
          {uniqueWordBucket && (
            <button type="button" className="browse-chip" onClick={clearUniqueWordBucket}>
              Unique words: {uniqueWordBucket} ×
            </button>
          )}
          {fameUnscoredOnly && (
            <button type="button" className="browse-chip" onClick={clearFameUnscoredOnly}>
              Fame: Not yet scored ×
            </button>
          )}
          {overallDifficultyBand && (
            <button type="button" className="browse-chip" onClick={clearOverallDifficultyBand}>
              Difficulty: {overallDifficultyBand} ×
            </button>
          )}
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      <ul className="authors-list">
        {items.map((a) => {
          const density = densityLabel(a.density)
          const overall = overallDifficultyLabel(a.overall_difficulty)
          return (
            <li
              key={a.author}
              className="authors-row"
              onClick={() => navigate(`/app/authors/${encodeURIComponent(a.author)}`)}
            >
              <span className="authors-name">{a.author}</span>
              <span className="authors-counts">
                {a.book_count} {a.book_count === 1 ? 'work' : 'works'} · {a.word_count}{' '}
                {a.word_count === 1 ? 'word' : 'words'}
                {a.mean_difficulty != null && <span> · {a.mean_difficulty.toFixed(1)} avg. difficulty</span>}
                {density && <span> · {density}</span>}
                {overall && <span> · {overall}</span>}
                {sort === 'unique_word_count' && (
                  <span>
                    {' '}
                    · {a.unique_word_count} unique {a.unique_word_count === 1 ? 'word' : 'words'}
                  </span>
                )}
                {a.fame_score != null && (
                  <span title={a.fame_reasoning || ''}> · ⭐ {a.fame_score.toFixed(1)}</span>
                )}
              </span>
            </li>
          )
        })}
        {!loading && items.length === 0 && <li className="authors-empty">No authors match.</li>}
      </ul>

      <Pagination page={page} totalPages={totalPages} total={total} itemLabel="authors" onPageChange={setPage} />
    </div>
  )
}

export default Authors
