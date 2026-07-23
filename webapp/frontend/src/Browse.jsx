import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from './AuthContext'
import AuthorSelect from './AuthorSelect'
import MultiSelect from './MultiSelect'
import { colorForBucket } from './domainColors'
import { usePagedTable } from './usePagedTable'
import './Browse.css'

const API_BASE = ''
const ARCHAIC_VALUES = ['current', 'dated', 'archaic', 'obsolete']
const ALPHABET = 'abcdefghijklmnopqrstuvwxyz'.split('')
const PAGE_SIZE = 30

// Every facet lives in the URL (useSearchParams), not local state -- a
// filtered combination like "archaic Nature & Science words in Ulysses" is
// exactly the kind of thing worth bookmarking or sharing on a personal
// reference tool, unlike the admin curation tables where that was never a
// real need. Page/sort/dir stay local view state via the shared
// usePagedTable hook (unmodified) rather than also URL-syncing those --
// the bookmarkable value is overwhelmingly in the facets, not the page number.
function Browse() {
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()
  const { user, logout } = useAuth()

  // Existing quick-jump search box -- untouched. Different intent ("I know
  // the word, take me there") from the faceted browse below ("let me explore").
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const searchRef = useRef(null)

  useEffect(() => {
    function handlePointerDown(e) {
      if (searchRef.current && !searchRef.current.contains(e.target)) setSuggestions([])
    }
    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [])

  useEffect(() => {
    if (!query.trim()) {
      setSuggestions([])
      return
    }
    const handle = setTimeout(() => {
      fetch(`${API_BASE}/api/words/search?q=${encodeURIComponent(query)}&limit=15`)
        .then((res) => res.json())
        .then(setSuggestions)
        .catch(() => {})
    }, 200)
    return () => clearTimeout(handle)
  }, [query])

  // --- facets, derived from the URL ------------------------------------------
  const author = searchParams.get('author')
  const bookIds = searchParams.getAll('book_id')
  const domains = searchParams.getAll('domain')
  const difficultyMin = searchParams.get('difficulty_min') || ''
  const difficultyMax = searchParams.get('difficulty_max') || ''
  const archaic = searchParams.getAll('archaic')
  const quizzableOnly = searchParams.get('quizzable_only') === 'true'
  const letter = searchParams.get('letter')

  function updateParams(mutator) {
    const next = new URLSearchParams(searchParams)
    mutator(next)
    setSearchParams(next)
  }

  function setAuthor(value) {
    updateParams((next) => {
      if (value) next.set('author', value)
      else next.delete('author')
      // Book choice is scoped to an author's own list -- clear it when the
      // author changes so the shelf never shows a book chip silently
      // orphaned from a different writer.
      next.delete('book_id')
    })
  }

  function setBookIds(ids) {
    updateParams((next) => {
      next.delete('book_id')
      ids.forEach((id) => next.append('book_id', id))
    })
  }

  function toggleDomain(bucket) {
    updateParams((next) => {
      const current = next.getAll('domain')
      next.delete('domain')
      const nextList = current.includes(bucket) ? current.filter((d) => d !== bucket) : [...current, bucket]
      nextList.forEach((d) => next.append('domain', d))
    })
  }

  function setDifficultyRange(min, max) {
    updateParams((next) => {
      if (min !== '' && min != null) next.set('difficulty_min', min)
      else next.delete('difficulty_min')
      if (max !== '' && max != null) next.set('difficulty_max', max)
      else next.delete('difficulty_max')
    })
  }

  function toggleArchaic(value) {
    updateParams((next) => {
      const current = next.getAll('archaic')
      next.delete('archaic')
      const nextList = current.includes(value) ? current.filter((a) => a !== value) : [...current, value]
      nextList.forEach((a) => next.append('archaic', a))
    })
  }

  function setQuizzableOnly(value) {
    updateParams((next) => {
      if (value) next.set('quizzable_only', 'true')
      else next.delete('quizzable_only')
    })
  }

  function setLetter(value) {
    updateParams((next) => {
      if (value) next.set('letter', value)
      else next.delete('letter')
    })
  }

  function clearAll() {
    setSearchParams(new URLSearchParams())
  }

  const activeFacetCount =
    (author ? 1 : 0) + bookIds.length + domains.length +
    (difficultyMin || difficultyMax ? 1 : 0) + archaic.length +
    (quizzableOnly ? 1 : 0) + (letter ? 1 : 0)

  // Every dependent fetch below (books, domain counts, difficulty bands) is
  // fully derived from `searchParams` -- using its own string form as the
  // effect dependency avoids re-deriving separate join(',') keys per array.
  const paramsKey = searchParams.toString()

  function facetSearchParams(extra = {}) {
    const p = new URLSearchParams()
    if (author) p.set('author', author)
    bookIds.forEach((id) => p.append('book_id', id))
    domains.forEach((d) => p.append('domain', d))
    if (difficultyMin !== '') p.set('difficulty_min', difficultyMin)
    if (difficultyMax !== '') p.set('difficulty_max', difficultyMax)
    archaic.forEach((a) => p.append('archaic', a))
    if (quizzableOnly) p.set('quizzable_only', 'true')
    if (letter) p.set('letter', letter)
    for (const [k, v] of Object.entries(extra)) p.set(k, v)
    return p
  }

  // --- book facet: MultiSelect over titles, scoped to the chosen author ------
  const [bookOptions, setBookOptions] = useState([])
  useEffect(() => {
    const params = new URLSearchParams({ page_size: '200', sort: 'word_count', dir: 'desc' })
    if (author) params.set('author', author)
    fetch(`${API_BASE}/api/browse/books?${params}`)
      .then((res) => res.json())
      .then((data) => setBookOptions(data.items))
      .catch(() => {})
  }, [author])

  const bookTitleToId = new Map(bookOptions.map((b) => [b.title, String(b.id)]))
  const bookIdToTitle = new Map(bookOptions.map((b) => [String(b.id), b.title]))
  const selectedBookTitles = bookIds.map((id) => bookIdToTitle.get(id)).filter(Boolean)

  function handleBookTitlesChange(titles) {
    setBookIds(titles.map((t) => bookTitleToId.get(t)).filter(Boolean))
  }

  // --- domain facet: chip row with live, other-facet-conditioned counts -----
  const [domainCounts, setDomainCounts] = useState([])
  useEffect(() => {
    fetch(`${API_BASE}/api/browse/domains?${facetSearchParams()}`)
      .then((res) => res.json())
      .then(setDomainCounts)
      .catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paramsKey])

  const bucketName = Object.fromEntries(domainCounts.map((d) => [d.bucket, d.name]))

  // --- difficulty facet: dual input + a band histogram -----------------------
  const [bands, setBands] = useState([])
  useEffect(() => {
    fetch(`${API_BASE}/api/browse/difficulty-bands?${facetSearchParams()}`)
      .then((res) => res.json())
      .then(setBands)
      .catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paramsKey])

  const bandsTotal = bands.reduce((sum, b) => sum + b.word_count, 0) || 1

  function surpriseMe() {
    fetch(`${API_BASE}/api/browse/words?${facetSearchParams({ random: 'true' })}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.items[0]) navigate(`/app/words/${data.items[0].id}`)
      })
      .catch(() => {})
  }

  // --- results -----------------------------------------------------------------
  const { items, total, page, setPage, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/browse/words',
    pageSize: PAGE_SIZE,
    defaultSort: 'lemma',
    defaultDir: 'asc',
    extraParams: {
      author: author || '',
      book_id: bookIds,
      domain: domains,
      difficulty_min: difficultyMin,
      difficulty_max: difficultyMax,
      archaic,
      quizzable_only: quizzableOnly,
      letter: letter || '',
    },
  })

  return (
    <div className="browse-page">
      <header className="browse-header">
        <h1>Vocab Browse</h1>
        <div className="browse-user">
          <Link to="/app/authors" className="browse-quiz-link">Browse by author</Link>
          {' · '}
          <Link to="/app/visualizations" className="browse-quiz-link">Visualizations</Link>
          {' · '}
          <Link to="/app/quiz" className="browse-quiz-link">Take a quiz</Link>
          {' · '}
          {user?.username} · <button type="button" onClick={logout}>Log out</button>
        </div>
      </header>

      <div className="browse-search" ref={searchRef}>
        <input
          type="text"
          placeholder="Search for a word…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {suggestions.length > 0 && (
          <ul className="browse-suggestions">
            {suggestions.map((w) => (
              <li key={w.id} onClick={() => navigate(`/app/words/${w.id}`)}>
                {w.lemma}
              </li>
            ))}
          </ul>
        )}
      </div>

      <section className="browse-facets">
        <div className="browse-facet-row">
          <AuthorSelect value={author} onChange={setAuthor} />
          <MultiSelect
            label="Book"
            options={bookOptions.map((b) => b.title)}
            selected={selectedBookTitles}
            onChange={handleBookTitlesChange}
            searchable
          />
          <label className="browse-checkbox">
            <input type="checkbox" checked={quizzableOnly} onChange={(e) => setQuizzableOnly(e.target.checked)} />
            Quizzable only
          </label>
          <button type="button" className="browse-surprise" onClick={surpriseMe}>
            🎲 Surprise me
          </button>
        </div>

        <div className="browse-domain-row">
          {domainCounts.map((d) => (
            <button
              type="button"
              key={d.bucket}
              className={domains.includes(d.bucket) ? 'browse-domain-chip active' : 'browse-domain-chip'}
              style={domains.includes(d.bucket) ? { borderColor: colorForBucket(d.bucket) } : undefined}
              onClick={() => toggleDomain(d.bucket)}
            >
              <span className="browse-domain-swatch" style={{ background: colorForBucket(d.bucket) }} />
              {d.name} <span className="browse-domain-count">{d.word_count}</span>
            </button>
          ))}
        </div>

        <div className="browse-difficulty-row">
          <span className="browse-facet-label">Difficulty</span>
          <input
            type="number" min="0" max="100" placeholder="0"
            value={difficultyMin}
            onChange={(e) => setDifficultyRange(e.target.value, difficultyMax)}
          />
          <span>–</span>
          <input
            type="number" min="0" max="100" placeholder="100"
            value={difficultyMax}
            onChange={(e) => setDifficultyRange(difficultyMin, e.target.value)}
          />
          <div className="browse-difficulty-histogram">
            {bands.map((b) => (
              <div
                key={b.label}
                className={b.band_min === null ? 'browse-band unscored' : 'browse-band'}
                style={{ flexGrow: Math.max(b.word_count, 1) / bandsTotal }}
                title={`${b.label}: ${b.word_count} words`}
                onClick={() => b.band_min !== null && setDifficultyRange(b.band_min, b.band_max)}
              />
            ))}
          </div>
        </div>

        <div className="browse-archaic-row">
          <span className="browse-facet-label">Archaic-ness</span>
          {ARCHAIC_VALUES.map((v) => (
            <button
              type="button"
              key={v}
              className={archaic.includes(v) ? 'browse-pill active' : 'browse-pill'}
              onClick={() => toggleArchaic(v)}
            >
              {v}
            </button>
          ))}
        </div>

        <div className="browse-az-strip">
          {ALPHABET.map((l) => (
            <button
              type="button"
              key={l}
              className={letter === l ? 'browse-az-letter active' : 'browse-az-letter'}
              onClick={() => setLetter(letter === l ? null : l)}
            >
              {l}
            </button>
          ))}
        </div>

        {activeFacetCount > 0 && (
          <div className="browse-shelf">
            {author && (
              <button type="button" className="browse-chip" onClick={() => setAuthor(null)}>
                Author: {author} ×
              </button>
            )}
            {bookIds.map((id) => (
              <button type="button" key={id} className="browse-chip"
                      onClick={() => setBookIds(bookIds.filter((b) => b !== id))}>
                Book: {bookIdToTitle.get(id) || `#${id}`} ×
              </button>
            ))}
            {domains.map((d) => (
              <button type="button" key={d} className="browse-chip" onClick={() => toggleDomain(d)}>
                Domain: {bucketName[d] || d} ×
              </button>
            ))}
            {(difficultyMin || difficultyMax) && (
              <button type="button" className="browse-chip" onClick={() => setDifficultyRange('', '')}>
                Difficulty: {difficultyMin || 0}–{difficultyMax || 100} ×
              </button>
            )}
            {archaic.map((a) => (
              <button type="button" key={a} className="browse-chip" onClick={() => toggleArchaic(a)}>
                {a} ×
              </button>
            ))}
            {quizzableOnly && (
              <button type="button" className="browse-chip" onClick={() => setQuizzableOnly(false)}>
                Quizzable only ×
              </button>
            )}
            {letter && (
              <button type="button" className="browse-chip" onClick={() => setLetter(null)}>
                Starts with "{letter.toUpperCase()}" ×
              </button>
            )}
            {activeFacetCount > 1 && (
              <button type="button" className="browse-clear-all" onClick={clearAll}>
                Clear all
              </button>
            )}
          </div>
        )}
      </section>

      {error && <div className="error-banner">{error}</div>}

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
        {!loading && items.length === 0 && <li className="browse-empty">No words match these filters.</li>}
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

export default Browse
