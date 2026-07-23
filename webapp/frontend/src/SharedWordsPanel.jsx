import { useEffect, useState } from 'react'
import './SharedWordsPanel.css'

const API_BASE = ''

// "The what" behind every relatedness score/shared-word-count ("the why")
// on this app: the actual overlapping rare vocabulary between two specific
// books or authors. No existing modal/overlay pattern anywhere else in the
// codebase to reuse -- this is the first one, kept deliberately minimal
// (backdrop + centered panel, Escape/backdrop-click/× to close) rather than
// pulling in a dialog library for one component.
function SharedWordsPanel({ fetchUrl, titleA, titleB, onClose }) {
  const [words, setWords] = useState(null) // null = loading, [] = loaded empty
  const [error, setError] = useState('')

  useEffect(() => {
    setWords(null)
    setError('')
    fetch(`${API_BASE}${fetchUrl}`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => setWords(data.shared_words))
      .catch((err) => setError(err.message || 'failed to load shared words'))
  }, [fetchUrl])

  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  return (
    <div className="shared-words-backdrop" onClick={onClose}>
      <div className="shared-words-panel" onClick={(e) => e.stopPropagation()}>
        <header className="shared-words-header">
          <h2>
            {titleA} <span className="shared-words-vs">×</span> {titleB}
          </h2>
          <button type="button" className="shared-words-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        {error && <div className="error-banner">{error}</div>}
        {words === null && !error && <p className="muted">Loading…</p>}
        {words !== null && words.length === 0 && (
          <p className="muted">No shared rare vocabulary found between these two.</p>
        )}
        {words !== null && words.length > 0 && (
          <>
            <p className="muted">{words.length} shared words, rarest first</p>
            <ul className="shared-words-list">
              {words.map((w) => (
                <li key={w.id}>
                  <span className="shared-words-lemma">{w.lemma}</span>
                  {w.definition && <span className="shared-words-def"> — {w.definition}</span>}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}

export default SharedWordsPanel
