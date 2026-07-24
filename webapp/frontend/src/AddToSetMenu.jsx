import { useEffect, useRef, useState } from 'react'
import './AddToSetMenu.css'

const API_BASE = ''

// Shared "add word(s) to a set" dropdown -- used both by Browse's bulk-select
// bar (wordIds = every checked row) and WordDetail's single-word button
// (wordIds = [that one id]). Modeled on AuthorSelect/Browse's own
// outside-click-close + debounce-free fetch-on-open pattern, not a new one.
function AddToSetMenu({ wordIds, label = 'Add to set' }) {
  const [open, setOpen] = useState(false)
  const [sets, setSets] = useState(null) // null = not loaded yet
  const [newName, setNewName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [justAdded, setJustAdded] = useState(false)
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
    setError('')
    fetch(`${API_BASE}/api/sets`)
      .then((res) => (res.ok ? res.json() : []))
      .then(setSets)
      .catch(() => setSets([]))
  }, [open])

  function flashAdded() {
    setOpen(false)
    setJustAdded(true)
    setTimeout(() => setJustAdded(false), 1500)
  }

  function addToSet(setId) {
    if (!wordIds.length) return
    setBusy(true)
    setError('')
    fetch(`${API_BASE}/api/sets/${setId}/words`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word_ids: wordIds }),
    })
      .then((res) => {
        if (!res.ok) throw new Error('failed to add to set')
        flashAdded()
      })
      .catch(() => setError('Could not add to set.'))
      .finally(() => setBusy(false))
  }

  function createAndAdd() {
    const name = newName.trim()
    if (!name || busy) return
    setBusy(true)
    setError('')
    fetch(`${API_BASE}/api/sets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    })
      .then((res) => {
        if (res.status === 409) throw new Error('You already have a set with that name.')
        if (!res.ok) throw new Error('failed to create set')
        return res.json()
      })
      .then((set) =>
        fetch(`${API_BASE}/api/sets/${set.id}/words`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ word_ids: wordIds }),
        }),
      )
      .then((res) => {
        if (!res.ok) throw new Error('failed to add to set')
        setNewName('')
        flashAdded()
      })
      .catch((err) => setError(err.message || 'Something went wrong.'))
      .finally(() => setBusy(false))
  }

  return (
    <div className="add-to-set-menu" ref={ref}>
      <button
        type="button"
        className="add-to-set-btn"
        onClick={() => setOpen((o) => !o)}
        disabled={!wordIds.length}
      >
        {justAdded ? 'Added ✓' : label}
      </button>
      {open && (
        <div className="add-to-set-dropdown">
          {sets === null && <p className="add-to-set-loading muted">Loading…</p>}
          {sets !== null && sets.length > 0 && (
            <ul className="add-to-set-list">
              {sets.map((s) => (
                <li key={s.id} onClick={() => addToSet(s.id)}>
                  <span className="add-to-set-name">{s.name}</span>
                  <span className="add-to-set-count">{s.word_count}</span>
                </li>
              ))}
            </ul>
          )}
          {sets !== null && sets.length === 0 && <p className="add-to-set-loading muted">No sets yet.</p>}
          <div className="add-to-set-new">
            <input
              type="text"
              placeholder="New set name…"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') createAndAdd()
              }}
            />
            <button type="button" onClick={createAndAdd} disabled={busy || !newName.trim()}>
              Create
            </button>
          </div>
          {error && <p className="add-to-set-error">{error}</p>}
        </div>
      )}
    </div>
  )
}

export default AddToSetMenu
