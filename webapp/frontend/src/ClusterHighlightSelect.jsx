import { useEffect, useRef, useState } from 'react'

// Highlight picker for AuthorsRelatedness/BooksRelatedness -- deliberately
// NOT AuthorSelect/BookSelect (which search the whole corpus): the map/
// matrix/dendrogram only ever show the top-N entities from their own
// clustering run, so letting someone search-and-pick from all ~3,500
// authors or ~11,000 books meant the overwhelming majority of picks landed
// on "isn't in this view's top entities" -- a highlight control that
// mostly doesn't work. `items` is the parent's already-fetched clustering-
// run node list (same one the page's own "Top N" header count comes from),
// filtered/searched entirely client-side -- at ~200 entries there's no
// reason to round-trip to the server for this.
function ClusterHighlightSelect({ items, value, onChange, placeholder = 'Search…' }) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    function handlePointerDown(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [])

  const selected = value != null ? items.find((it) => it.id === value) : null
  const q = query.trim().toLowerCase()
  const filtered = (q ? items.filter((it) => it.label.toLowerCase().includes(q)) : items).slice(0, 30)

  function pick(item) {
    onChange(item.id)
    setQuery('')
    setOpen(false)
  }

  function clear() {
    onChange(null)
    setQuery('')
  }

  return (
    <div className="author-select" ref={ref}>
      {selected ? (
        <button type="button" className="author-select-chosen" onClick={clear}>
          {selected.label} <span className="author-select-clear">×</span>
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
      {open && !selected && filtered.length > 0 && (
        <ul className="author-select-list">
          {filtered.map((it) => (
            <li key={it.id} onClick={() => pick(it)}>
              <span className="author-select-name">{it.label}</span>
              {it.sublabel && <span className="author-select-count">{it.sublabel}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default ClusterHighlightSelect
