import { useCallback, useEffect, useState } from 'react'
import './App.css'

const API_BASE = '' // relative — dev proxies /api via vite.config.js, prod is same-origin
const PAGE_SIZE = 50

const COLUMNS = [
  { key: 'lemma', label: 'Term' },
  { key: 'part_of_speech', label: 'POS' },
  { key: 'definition', label: 'Definition' },
  { key: 'difficulty', label: 'Difficulty' },
]

function App() {
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [sort, setSort] = useState('difficulty')
  const [dir, setDir] = useState('asc')
  const [pos, setPos] = useState('')
  const [posOptions, setPosOptions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/pos-values`)
      .then((res) => res.json())
      .then(setPosOptions)
      .catch(() => {})
  }, [])

  const load = useCallback(() => {
    setLoading(true)
    setError(null)
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(PAGE_SIZE),
      sort,
      dir,
    })
    if (pos) params.set('pos', pos)
    fetch(`${API_BASE}/api/words?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error(`load failed: ${res.status}`)
        return res.json()
      })
      .then((data) => {
        setItems(data.items)
        setTotal(data.total)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [page, sort, dir, pos])

  useEffect(() => {
    load()
  }, [load])

  function handlePosChange(value) {
    setPos(value)
    setPage(1)
  }

  function handleSort(key) {
    if (sort === key) {
      setDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSort(key)
      setDir('asc')
    }
    setPage(1)
  }

  function handleDelete(id) {
    setItems((prev) => prev.filter((w) => w.id !== id))
    setTotal((t) => t - 1)
    fetch(`${API_BASE}/api/words/${id}`, { method: 'DELETE' }).catch(() => {
      setError(`failed to delete word ${id} — reload to resync`)
      load()
    })
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="review-app">
      <header>
        <h1>Vocab Review</h1>
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
        </div>
      </header>

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
    </div>
  )
}

export default App
