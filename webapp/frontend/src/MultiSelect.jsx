import { useEffect, useRef, useState } from 'react'

/** A checkbox-list dropdown for filtering by zero or more of a fixed option set.
 * `searchable` adds a text filter at the top of the panel -- needed once the
 * option list is too long to scan as a flat checkbox list (e.g. book titles
 * with no author selected yet). */
function MultiSelect({ label, options, selected, onChange, searchable = false }) {
  const [open, setOpen] = useState(false)
  const [filterText, setFilterText] = useState('')
  const ref = useRef(null)

  useEffect(() => {
    function handleClickOutside(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  useEffect(() => {
    if (!open) setFilterText('')
  }, [open])

  function toggle(value) {
    if (selected.includes(value)) {
      onChange(selected.filter((v) => v !== value))
    } else {
      onChange([...selected, value])
    }
  }

  const summary =
    selected.length === 0 ? 'All' : selected.length === 1 ? selected[0] : `${selected.length} selected`

  const visibleOptions = searchable && filterText
    ? options.filter((opt) => opt.toLowerCase().includes(filterText.toLowerCase()))
    : options

  return (
    <div className="multiselect" ref={ref}>
      <button type="button" className="multiselect-trigger" onClick={() => setOpen((o) => !o)}>
        {label}: {summary} <span className="multiselect-caret">{open ? '▴' : '▾'}</span>
      </button>
      {open && (
        <div className="multiselect-panel">
          {searchable && (
            <input
              type="text"
              className="multiselect-search"
              placeholder={`Search ${label.toLowerCase()}…`}
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              autoFocus
            />
          )}
          <label className="multiselect-option">
            <input type="checkbox" checked={selected.length === 0} onChange={() => onChange([])} />
            All
          </label>
          {visibleOptions.map((opt) => (
            <label key={opt} className="multiselect-option">
              <input type="checkbox" checked={selected.includes(opt)} onChange={() => toggle(opt)} />
              {opt}
            </label>
          ))}
          {searchable && visibleOptions.length === 0 && (
            <div className="multiselect-empty">No matches</div>
          )}
        </div>
      )}
    </div>
  )
}

export default MultiSelect
