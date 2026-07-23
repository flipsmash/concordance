import { useEffect, useRef, useState } from 'react'

const API_BASE = ''

// Single-select book typeahead, modeled directly on AuthorSelect.jsx (same
// debounce, same outside-click-close, same browsable-empty-handed default
// via /api/browse/books' own word_count-desc sort). Selects a full book
// object (not just an id/title) since callers need book.author too, to
// build a /app/authors/:author/:bookId/... link.
function BookSelect({ onPick, placeholder = 'Book…' }) {
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
    onPick(book)
    setQuery('')
    setOpen(false)
  }

  return (
    <div className="author-select" ref={ref}>
      <input
        type="text"
        className="author-select-input"
        placeholder={placeholder}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => setOpen(true)}
      />
      {open && suggestions.length > 0 && (
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
