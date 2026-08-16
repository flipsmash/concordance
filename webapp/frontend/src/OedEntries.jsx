import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import AlphabetStrip from './AlphabetStrip'
import OedMatchBadge from './OedMatchBadge'
import SortControl from './SortControl'
import { usePagedTable } from './usePagedTable'
import './Authors.css'
import './OedEntries.css'

const API_BASE = ''
const PAGE_SIZE = 50

const MATCH_FILTER_OPTIONS = [
  { value: '', label: 'Any concordance match' },
  { value: 'unchecked', label: 'Not yet checked' },
  { value: 'accepted', label: 'In concordance' },
  { value: 'pruned', label: 'Pruned from concordance' },
  { value: 'rejected', label: 'Rejected in concordance' },
  { value: 'unique', label: 'Not in concordance' },
]

const SORT_FIELDS = [
  { key: 'headword', label: 'Headword (A–Z)' },
  { key: 'page_number', label: 'Page number' },
  { key: 'created_at', label: 'Recently ingested' },
]

// Admin-only browse over the `oed` schema's ingested entries (see
// webapp/backend/oed.py) -- same list chrome as Books.jsx/Authors.jsx
// (shared Authors.css classes), since it's the identical browsing
// mechanism: search + letter jump + paginated row list, just against a
// different, standalone schema with no fame/difficulty facets of its own.
function OedEntries() {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [letter, setLetter] = useState(null)
  const [volumeId, setVolumeId] = useState('')
  const [needsReview, setNeedsReview] = useState(false)
  const [concordanceMatch, setConcordanceMatch] = useState('')
  const [volumes, setVolumes] = useState([])

  useEffect(() => {
    fetch(`${API_BASE}/api/admin/oed/volumes`)
      .then((res) => res.json())
      .then(setVolumes)
      .catch(() => {})
  }, [])

  useEffect(() => {
    const handle = setTimeout(() => setDebounced(query.trim()), 200)
    return () => clearTimeout(handle)
  }, [query])

  const { items, total, page, setPage, sort, dir, handleSort, loading, error, totalPages } = usePagedTable({
    endpoint: '/api/admin/oed/entries',
    pageSize: PAGE_SIZE,
    defaultSort: 'headword',
    defaultDir: 'asc',
    extraParams: {
      q: debounced, volume_id: volumeId, letter,
      needs_review: needsReview || undefined,
      concordance_match: concordanceMatch || undefined,
    },
  })

  function changeLetter(l) {
    setLetter(l)
    setPage(1)
  }

  function changeVolume(v) {
    setVolumeId(v)
    setPage(1)
  }

  function changeNeedsReview(v) {
    setNeedsReview(v)
    setPage(1)
  }

  function changeConcordanceMatch(v) {
    setConcordanceMatch(v)
    setPage(1)
  }

  return (
    <div className="authors-page">
      <header className="authors-header">
        <h1>OED entries</h1>
      </header>

      <input
        type="text"
        className="authors-search"
        placeholder="Search headwords…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        autoFocus
      />

      <div className="authors-toolbar">
        <AlphabetStrip letter={letter} onChange={changeLetter} />
        <select value={volumeId} onChange={(e) => changeVolume(e.target.value)}>
          <option value="">All volumes</option>
          {volumes.map((v) => (
            <option key={v.id} value={v.id}>{v.volume_label || v.file_name}</option>
          ))}
        </select>
        <label className="oed-review-filter">
          <input
            type="checkbox"
            checked={needsReview}
            onChange={(e) => changeNeedsReview(e.target.checked)}
          />
          Needs pronunciation review
        </label>
        <select value={concordanceMatch} onChange={(e) => changeConcordanceMatch(e.target.value)}>
          {MATCH_FILTER_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        <SortControl fields={SORT_FIELDS} sort={sort} dir={dir} onSort={handleSort} />
      </div>

      {error && <div className="error-banner">{error}</div>}

      <ul className="authors-list">
        {items.map((e) => (
          <li
            key={e.id}
            className="authors-row work-row"
            onClick={() => navigate(`${e.id}`)}
          >
            <span className="work-title">
              {e.headword}
              {e.homograph_number != null && <sup>{e.homograph_number}</sup>}
            </span>
            {e.first_definition ? (
              <span className="oed-def-preview">
                {e.first_definition.length > 140
                  ? `${e.first_definition.slice(0, 140)}…`
                  : e.first_definition}
              </span>
            ) : (
              <span className="oed-def-preview oed-no-def">no definition parsed</span>
            )}
            <span className="work-stats">
              {e.part_of_speech && <span>{e.part_of_speech}</span>}
              <span className="work-count">p. {e.page_number}</span>
              {e.pronunciation_ipa
                ? <span className="oed-ipa">/{e.pronunciation_ipa}/</span>
                : <span className="oed-no-ipa">no pronunciation</span>}
              {e.pronunciation_needs_review && (
                <span className="oed-review-badge">needs review</span>
              )}
              <OedMatchBadge match={e.concordance_match} />
            </span>
          </li>
        ))}
        {!loading && items.length === 0 && <li className="authors-empty">No entries match.</li>}
      </ul>

      <footer className="authors-footer">
        <button type="button" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
          ← Prev
        </button>
        <span>
          Page {page} of {totalPages} ({total} entries)
        </span>
        <button type="button" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
          Next →
        </button>
      </footer>
    </div>
  )
}

export default OedEntries
