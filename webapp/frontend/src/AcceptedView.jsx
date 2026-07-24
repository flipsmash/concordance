import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import InviteButton from './InviteButton'
import { usePagedTable } from './usePagedTable'
import './Browse.css' // .browse-az-strip/.browse-az-letter -- reused here as-is

const API_BASE = ''
const PAGE_SIZE = 50
const ALPHABET = 'abcdefghijklmnopqrstuvwxyz'.split('')

const COLUMNS = [
  { key: 'lemma', label: 'Term' },
  { key: 'part_of_speech', label: 'POS' },
  { key: 'definition', label: 'Definition' },
  { key: 'difficulty', label: 'Difficulty' },
]

function AcceptedView() {
  const [pos, setPos] = useState('')
  const [posOptions, setPosOptions] = useState([])
  const [qInput, setQInput] = useState('')
  const [q, setQ] = useState('')
  const [letter, setLetter] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/api/pos-values`)
      .then((res) => res.json())
      .then(setPosOptions)
      .catch(() => {})
  }, [])

  // Same 200ms debounce as Browse.jsx's own search box -- one keystroke
  // pause convention everywhere in the app, not a second, differently-timed one.
  useEffect(() => {
    const handle = setTimeout(() => setQ(qInput.trim()), 200)
    return () => clearTimeout(handle)
  }, [qInput])

  const {
    items, setItems, total, setTotal, page, setPage, sort, dir, handleSort,
    loading, error, setError, load, resetPage, totalPages,
  } = usePagedTable({
    endpoint: '/api/words',
    pageSize: PAGE_SIZE,
    defaultSort: 'difficulty',
    defaultDir: 'asc',
    extraParams: { pos, q, letter },
  })

  function handlePosChange(value) {
    setPos(value)
    resetPage()
  }

  function handleQChange(value) {
    setQInput(value)
    resetPage()
  }

  function handleLetterChange(value) {
    setLetter((current) => (current === value ? '' : value))
    resetPage()
  }

  function handleDelete(id) {
    setItems((prev) => prev.filter((w) => w.id !== id))
    setTotal((t) => t - 1)
    fetch(`${API_BASE}/api/words/${id}`, { method: 'DELETE' }).catch(() => {
      setError(`failed to delete word ${id} — reload to resync`)
      load()
    })
  }

  return (
    <>
      <div className="header-controls">
        <label className="pos-filter">
          POS:{' '}
          <select value={pos} onChange={(e) => handlePosChange(e.target.value)}>
            <option value="">All</option>
            {posOptions.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <span className="count">
          {total.toLocaleString()} active term{total === 1 ? '' : 's'}
        </span>
        <InviteButton />
      </div>

      <div className="browse-search">
        <input
          type="text"
          placeholder="Search for a word…"
          value={qInput}
          onChange={(e) => handleQChange(e.target.value)}
        />
      </div>

      <div className="browse-az-strip">
        {ALPHABET.map((l) => (
          <button
            type="button"
            key={l}
            className={letter === l ? 'browse-az-letter active' : 'browse-az-letter'}
            onClick={() => handleLetterChange(l)}
          >
            {l}
          </button>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th key={col.key} onClick={() => handleSort(col.key)} className="sortable">
                  {col.label}
                  {sort === col.key && <span className="arrow">{dir === 'asc' ? ' ▲' : ' ▼'}</span>}
                </th>
              ))}
              <th></th>
            </tr>
          </thead>
          <tbody>
            {items.map((w) => (
              <tr key={w.id}>
                <td className="lemma">
                  <Link to={`/words/${w.id}`}>{w.lemma}</Link>
                  {w.rescued_from_reject && (
                    <span className="rescued-badge" title="Added back from the Rejected tab">
                      rescued
                    </span>
                  )}
                </td>
                <td className="pos">{w.part_of_speech || '—'}</td>
                <td className="definition">{w.definition || '—'}</td>
                <td className="difficulty">{w.difficulty != null ? Math.round(w.difficulty) : '—'}</td>
                <td className="actions">
                  <button type="button" className="delete-btn" onClick={() => handleDelete(w.id)}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
            {!loading && items.length === 0 && (
              <tr>
                <td colSpan={5} className="empty">
                  Nothing here.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <footer>
        <button type="button" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
          ← Prev
        </button>
        <span>
          Page {page} of {totalPages}
        </span>
        <button type="button" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
          Next →
        </button>
      </footer>
    </>
  )
}

export default AcceptedView
