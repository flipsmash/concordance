import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './Sets.css'

const API_BASE = ''

// Landing page for user-curated word sets (§ word sets) -- companion to
// Browse/Authors, same "flat list, click a row to drill in" shape as
// Authors.jsx, just without usePagedTable's pagination: sets are user-
// curated, realistically a handful per person, not thousands like authors.
function Sets() {
  const navigate = useNavigate()
  const [sets, setSets] = useState(null) // null = loading
  const [error, setError] = useState('')
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)

  function load() {
    fetch(`${API_BASE}/api/sets`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to load sets'))))
      .then(setSets)
      .catch((err) => setError(err.message))
  }

  useEffect(load, [])

  function createSet() {
    const name = newName.trim()
    if (!name || creating) return
    setCreating(true)
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
      .then((set) => navigate(`/app/sets/${set.id}`))
      .catch((err) => setError(err.message))
      .finally(() => setCreating(false))
  }

  function deleteSet(e, setId) {
    e.stopPropagation()
    if (!window.confirm('Delete this set? This cannot be undone.')) return
    fetch(`${API_BASE}/api/sets/${setId}`, { method: 'DELETE' })
      .then((res) => {
        if (!res.ok) throw new Error('failed to delete set')
        setSets((prev) => prev.filter((s) => s.id !== setId))
      })
      .catch((err) => setError(err.message))
  }

  return (
    <div className="sets-page">
      <header className="sets-header">
        <h1>My word sets</h1>
      </header>

      <div className="sets-new-row">
        <input
          type="text"
          placeholder="New set name…"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') createSet()
          }}
        />
        <button type="button" onClick={createSet} disabled={creating || !newName.trim()}>
          Create set
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {sets === null && !error && <p className="muted">Loading…</p>}

      {sets !== null && sets.length === 0 && (
        <p className="sets-empty muted">
          No sets yet — create one above, or add words to a new set from Browse or a word's own page.
        </p>
      )}

      {sets !== null && sets.length > 0 && (
        <ul className="sets-list">
          {sets.map((s) => (
            <li key={s.id} className="sets-row" onClick={() => navigate(`/app/sets/${s.id}`)}>
              <span className="sets-name">{s.name}</span>
              <span className="sets-counts">
                {s.mastered_count}/{s.word_count} mastered
              </span>
              <button type="button" className="sets-delete-btn" onClick={(e) => deleteSet(e, s.id)}>
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default Sets
