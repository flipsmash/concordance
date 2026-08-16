import { useEffect, useRef, useState } from 'react'

// Multi-select typeahead: pick any number of items from a searched list,
// each shown as a removable chip. Generalizes AuthorSelect/BookSelect's
// debounced-fetch/dropdown/outside-click-close shape (same 200ms debounce,
// same outside-click-close ref pattern) to "add to a list" instead of
// "replace a single value" -- built fresh rather than reusing those two
// since their exported value shape (a bare author string / a full book
// object) is single-selection-specific throughout their existing callers;
// this one always deals in {id, label, sublabel?} triples regardless of
// what fetchItems pulls them from, so it works for authors and books alike.
//
// `fetchItems(query)` -> Promise<Array<{id, label, sublabel?}>>, called
// (200ms debounced) whenever the box is open and the query changes.
// `selected`/`onChange` hold the full list of {id, label} pairs, not just
// ids, so a chip can render its own label without a separate id->label
// lookup on the caller's side.
function TypeaheadMultiChips({ label, placeholder, fetchItems, selected, onChange }) {
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
      fetchItems(query.trim()).then(setSuggestions).catch(() => {})
    }, 200)
    return () => clearTimeout(handle)
  }, [query, open, fetchItems])

  const selectedIds = new Set(selected.map((s) => s.id))

  function pick(item) {
    if (!selectedIds.has(item.id)) onChange([...selected, item])
    setQuery('')
  }

  function remove(id) {
    onChange(selected.filter((s) => s.id !== id))
  }

  const visibleSuggestions = suggestions.filter((s) => !selectedIds.has(s.id))

  return (
    <div className="typeahead-multi" ref={ref}>
      {label && <span className="typeahead-multi-label">{label}</span>}
      <div className="typeahead-multi-chips">
        {selected.map((s) => (
          <button type="button" key={s.id} className="author-select-chosen" onClick={() => remove(s.id)}>
            {s.label} <span className="author-select-clear">×</span>
          </button>
        ))}
        <input
          type="text"
          className="author-select-input"
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setOpen(true)}
        />
      </div>
      {open && visibleSuggestions.length > 0 && (
        <ul className="author-select-list">
          {visibleSuggestions.map((s) => (
            <li key={s.id} onClick={() => pick(s)}>
              <span className="author-select-name">{s.label}</span>
              {s.sublabel != null && <span className="author-select-count">{s.sublabel}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default TypeaheadMultiChips
