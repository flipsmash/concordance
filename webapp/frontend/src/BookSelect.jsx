import { useEffect, useRef, useState } from 'react'

const API_BASE = ''

// Single-select book typeahead, modeled directly on AuthorSelect.jsx (same
// debounce, same outside-click-close, same browsable-empty-handed default
// via /api/browse/books' own word_count-desc sort). Selects a full book
// object (not just an id/title) since callers need book.author too, to
// build a /app/authors/:author/:bookId/... link.
//
// Two calling conventions, same as this component originally had one of:
// `onPick` (fire-and-forget -- Visualizations.jsx's "jump to this book's
// ego graph", clears back to an empty box after picking) or a controlled
// `value`/`onChange` pair (BooksRelatedness.jsx's "highlight" control,
// mirroring AuthorSelect's chosen-chip persistence across tab switches --
// the picked book stays visible with a clear (×) button until explicitly
// cleared). Both can be passed at once if a caller wants both effects.
function BookSelect({ onPick, value, onChange, placeholder = 'Book…' }) {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    function handlePointerDown(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [])

  useEffect(() => {
    if (!open) return
    const handle = setTimeout(() => {
      const params = new URLSearchParams({ page_size: '20', sort: 'word_count', dir: 'desc' })
      if (query.trim()) params.set('q', query.trim())
      fetch(`${API_BASE}/api/browse/books?${params}`)
        .then((res) => res.json())
        .then((data) => setSuggestions(data.items))
        .catch(() => {})
    }, 200)
    return () => clearTimeout(handle)
  }, [query, open])

  function pick(book) {
    onPick?.(book)
    onChange?.(book)
    setQuery('')
    setOpen(false)
  }

  function clear() {
    onChange?.(null)
    setQuery('')
  }

  return (
    <div className="author-select" ref={ref}>
      {value ? (
        <button type="button" className="author-select-chosen" onClick={clear}>
          {value.title} <span className="author-select-clear">×</span>
        </button>
      ) : (
        <input
          type="text"
          className="author-select-input"
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setOpen(true)}
        />
      )}
      {open && !value && suggestions.length > 0 && (
        <ul className="author-select-list">
          {suggestions.map((b) => (
            <li key={b.id} onClick={() => pick(b)}>
              <span className="author-select-name">{b.title}</span>
              <span className="author-select-count">{b.word_count}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default BookSelect
