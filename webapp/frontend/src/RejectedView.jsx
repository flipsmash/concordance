import { useEffect, useState } from 'react'
import { usePagedTable } from './usePagedTable'

const API_BASE = ''
const PAGE_SIZE = 50

const COLUMNS = [
  { key: 'lemma', label: 'Term' },
  { key: 'book', label: 'Book' },
  { key: 'reason', label: 'Reason' },
  { key: 'count', label: 'Count' },
  { key: 'zipf', label: 'Zipf' },
]

function RejectedView() {
  const [book, setBook] = useState('')
  const [bookOptions, setBookOptions] = useState([])
  const [addedIds, setAddedIds] = useState({})

  useEffect(() => {
    fetch(`${API_BASE}/api/rejected/books`)
      .then((res) => res.json())
      .then(setBookOptions)
      .catch(() => {})
  }, [])

  const {
    items, setItems, total, setTotal, page, setPage, sort, dir, handleSort,
    loading, error, setError, load, resetPage, totalPages,
  } = usePagedTable({
    endpoint: '/api/rejected',
    pageSize: PAGE_SIZE,
    defaultSort: 'count',
    defaultDir: 'desc',
    extraParams: { book },
  })

  function handleBookChange(value) {
    setBook(value)
    resetPage()
  }

  function handleAccept(id, lemma) {
    setAddedIds((prev) => ({ ...prev, [id]: 'adding' }))
    fetch(`${API_BASE}/api/rejected/${id}/accept`, { method: 'POST' })
      .then((res) => {
        if (!res.ok) throw new Error(`accept failed: ${res.status}`)
        return res.json()
      })
      .then(() => {
        setItems((prev) => prev.filter((w) => w.id !== id))
        setTotal((t) => t - 1)
      })
      .catch(() => {
        setError(`failed to accept "${lemma}" — reload to resync`)
        setAddedIds((prev) => {
          const next = { ...prev }
          delete next[id]
          return next
        })
        load()
      })
  }

  return (
    <>
      <div className="header-controls">
        <label className="pos-filter">
          Book:{' '}
          <select value={book} onChange={(e) => handleBookChange(e.target.value)}>
            <option value="">All</option>
            {bookOptions.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </label>
        <span className="count">
          {total.toLocaleString()} rejected term{total === 1 ? '' : 's'}
        </span>
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
                <td className="lemma">{w.lemma}</td>
                <td className="pos">{w.book}</td>
                <td className="reason" title={w.detail || undefined}>
                  {w.reason || '—'}
                </td>
                <td className="difficulty">{w.count ?? '—'}</td>
                <td className="difficulty">{w.zipf != null ? w.zipf.toFixed(2) : '—'}</td>
                <td className="actions">
                  <button
                    type="button"
                    className="accept-btn"
                    disabled={addedIds[w.id] === 'adding'}
                    onClick={() => handleAccept(w.id, w.lemma)}
                  >
                    {addedIds[w.id] === 'adding' ? 'Adding…' : 'Add'}
                  </button>
                </td>
              </tr>
            ))}
            {!loading && items.length === 0 && (
              <tr>
                <td colSpan={6} className="empty">
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

export default RejectedView
