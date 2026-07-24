import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import './SetDetail.css'

const API_BASE = ''

// Per-set summary page (§ word sets) -- every word in the set, its
// definition, and mastered status, per the feature's own spec. Also the
// entry point into a flashcard run and where mastery gets fixed up without
// redoing one (the same PATCH the flashcard-run "Mastered" button calls).
function SetDetail() {
  const { setId } = useParams()
  const navigate = useNavigate()
  const [set, setSet] = useState(null) // {id, name, items} | null (loading)
  const [error, setError] = useState('')
  const [renaming, setRenaming] = useState(false)
  const [nameInput, setNameInput] = useState('')

  function load() {
    fetch(`${API_BASE}/api/sets/${setId}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to load set'))))
      .then((data) => {
        setSet(data)
        setNameInput(data.name)
      })
      .catch((err) => setError(err.message))
  }

  useEffect(load, [setId])

  function saveRename() {
    const name = nameInput.trim()
    if (!name || name === set.name) {
      setRenaming(false)
      return
    }
    fetch(`${API_BASE}/api/sets/${setId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    })
      .then((res) => {
        if (res.status === 409) throw new Error('You already have a set with that name.')
        if (!res.ok) throw new Error('failed to rename set')
        return res.json()
      })
      .then((updated) => {
        setSet((prev) => ({ ...prev, name: updated.name }))
        setRenaming(false)
      })
      .catch((err) => setError(err.message))
  }

  function toggleMastered(wordId, mastered) {
    setSet((prev) => ({
      ...prev,
      items: prev.items.map((item) => (item.word_id === wordId ? { ...item, mastered } : item)),
    }))
    fetch(`${API_BASE}/api/sets/${setId}/words/${wordId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mastered }),
    }).catch(() => {
      setError('Failed to update — reload to resync')
      load()
    })
  }

  function removeWord(wordId) {
    setSet((prev) => ({ ...prev, items: prev.items.filter((item) => item.word_id !== wordId) }))
    fetch(`${API_BASE}/api/sets/${setId}/words/${wordId}`, { method: 'DELETE' }).catch(() => {
      setError('Failed to remove — reload to resync')
      load()
    })
  }

  if (error) return <div className="error-banner">{error}</div>
  if (!set) return <div className="page-loading">Loading…</div>

  const unmastered = set.items.filter((item) => !item.mastered).length

  return (
    <div className="set-detail-page">
      <header className="set-detail-header">
        <div>
          {renaming ? (
            <div className="set-detail-rename-row">
              <input
                type="text"
                value={nameInput}
                onChange={(e) => setNameInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') saveRename()
                  if (e.key === 'Escape') setRenaming(false)
                }}
                autoFocus
              />
              <button type="button" onClick={saveRename}>Save</button>
            </div>
          ) : (
            <h1 onClick={() => setRenaming(true)} title="Click to rename">
              {set.name}
            </h1>
          )}
          <p className="muted">
            {set.items.length} word{set.items.length === 1 ? '' : 's'} ·{' '}
            {set.items.length - unmastered} mastered
          </p>
        </div>
        <Link to="/app/sets" className="set-detail-back-link">← My sets</Link>
      </header>

      {unmastered > 0 ? (
        <button
          type="button"
          className="set-detail-start-btn"
          onClick={() => navigate(`/app/sets/${setId}/flashcards`)}
        >
          Start flashcards ({unmastered} to review)
        </button>
      ) : (
        <p className="set-detail-all-mastered">
          {set.items.length > 0 ? '🎉 All words in this set are mastered.' : 'This set has no words yet.'}
        </p>
      )}

      <ul className="set-detail-list">
        {set.items.map((item) => (
          <li key={item.word_id} className="set-detail-row">
            <label className="set-detail-mastered">
              <input
                type="checkbox"
                checked={item.mastered}
                onChange={(e) => toggleMastered(item.word_id, e.target.checked)}
              />
            </label>
            <div className="set-detail-word">
              <span className="set-detail-lemma">{item.lemma}</span>
              <span className="set-detail-def">{item.definition || '—'}</span>
            </div>
            <button type="button" className="set-detail-remove-btn" onClick={() => removeWord(item.word_id)}>
              Remove
            </button>
          </li>
        ))}
        {set.items.length === 0 && (
          <li className="set-detail-empty">
            No words yet — add some from Browse or a word's own page.
          </li>
        )}
      </ul>
    </div>
  )
}

export default SetDetail
