import { useEffect, useRef, useState } from 'react'

const API_BASE = ''

// Single-select author typeahead for the browse facets. Modeled directly on
// Browse.jsx's own debounced search-dropdown (same 200ms debounce, same
// outside-click-close ref pattern) rather than sharing a component with it --
// this one selects INTO a filter (value/onChange, stays open on this page),
// Browse's own box navigates AWAY on pick. Browsable empty-handed: opening
// with no query fetches authors sorted by word_count desc (the endpoint's
// default sort), so it's a light discovery list, not just a search box.
function AuthorSelect({ value, onChange }) {
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
      const params = new URLSearchParams({ page_size: '20' })
      if (query.trim()) params.set('q', query.trim())
      fetch(`${API_BASE}/api/browse/authors?${params}`)
        .then((res) => res.json())
        .then((data) => setSuggestions(data.items))
        .catch(() => {})
    }, 200)
    return () => clearTimeout(handle)
  }, [query, open])

  function pick(author) {
    onChange(author)
    setQuery('')
    setOpen(false)
  }

  function clear() {
    onChange(null)
    setQuery('')
  }

  return (
    <div className="author-select" ref={ref}>
      {value ? (
        <button type="button" className="author-select-chosen" onClick={clear}>
          {value} <span className="author-select-clear">×</span>
        </button>
      ) : (
        <input
          type="text"
          className="author-select-input"
          placeholder="Author…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setOpen(true)}
        />
      )}
      {open && !value && suggestions.length > 0 && (
        <ul className="author-select-list">
          {suggestions.map((a) => (
            <li key={a.author} onClick={() => pick(a.author)}>
              <span className="author-select-name">{a.author}</span>
              <span className="author-select-count">{a.word_count}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default AuthorSelect
