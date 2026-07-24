import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import './FlashcardRun.css'
import './QuizRun.css' // .quiz-progress-track/.quiz-progress-fill/.quiz-progress-label -- reused here as-is

const API_BASE = ''

// Fisher-Yates -- unbiased shuffle, done once per deck load (see the
// module's own reasoning in the plan: no flashcard_session table, this is
// the client-side "randomly ordered" the whole run is stateless around).
function shuffle(array) {
  const copy = [...array]
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[copy[i], copy[j]] = [copy[j], copy[i]]
  }
  return copy
}

function FlashcardRun() {
  const { setId } = useParams()
  const [deck, setDeck] = useState(null) // shuffled array | null (loading)
  const [error, setError] = useState('')
  const [index, setIndex] = useState(0)
  const [flipped, setFlipped] = useState(false)
  const [masteredThisRun, setMasteredThisRun] = useState(0)

  useEffect(() => {
    fetch(`${API_BASE}/api/sets/${setId}/flashcards`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to load flashcards'))))
      .then((data) => setDeck(shuffle(data.items)))
      .catch((err) => setError(err.message))
  }, [setId])

  const card = deck && index < deck.length ? deck[index] : null

  function advance() {
    setFlipped(false)
    setIndex((i) => i + 1)
  }

  function markMastered() {
    if (!card) return
    setMasteredThisRun((n) => n + 1)
    fetch(`${API_BASE}/api/sets/${setId}/words/${card.word_id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mastered: true }),
    }).catch(() => {})
    advance()
  }

  // Space flips the card; Right arrow advances without marking mastered
  // ("still learning" case) -- see the plan's keyboard-shortcut section.
  useEffect(() => {
    function handleKeyDown(e) {
      if (!card) return
      if (e.code === 'Space') {
        e.preventDefault()
        setFlipped((f) => !f)
      } else if (e.code === 'ArrowRight') {
        e.preventDefault()
        advance()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [card])

  if (error) return <div className="error-banner">{error}</div>
  if (!deck) return <div className="page-loading">Loading…</div>

  if (deck.length === 0) {
    return (
      <div className="flashcard-run-page">
        <p className="flashcard-done-message">🎉 Nothing left to review in this set right now.</p>
        <Link to={`/app/sets/${setId}`} className="flashcard-done-link">← Back to set</Link>
      </div>
    )
  }

  if (!card) {
    return (
      <div className="flashcard-run-page">
        <p className="flashcard-done-message">
          Done! {masteredThisRun} word{masteredThisRun === 1 ? '' : 's'} marked mastered this run.
        </p>
        <div className="flashcard-done-links">
          <Link to={`/app/sets/${setId}`} className="flashcard-done-link">← Back to set summary</Link>
          <Link to="/app/sets" className="flashcard-done-link">My sets</Link>
        </div>
      </div>
    )
  }

  const progress = Math.round((index / deck.length) * 100)

  return (
    <div className="flashcard-run-page">
      <div className="quiz-progress-track">
        <div className="quiz-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <p className="quiz-progress-label">
        Card {index + 1} of {deck.length}
      </p>

      <div
        className={flipped ? 'flashcard flipped' : 'flashcard'}
        onClick={() => setFlipped((f) => !f)}
        role="button"
        tabIndex={0}
      >
        <div className="flashcard-inner">
          <div className="flashcard-face flashcard-front">{card.lemma}</div>
          <div className="flashcard-face flashcard-back">{card.definition || '—'}</div>
        </div>
      </div>
      <p className="flashcard-hint muted">Press space or click the card to flip</p>

      {flipped && (
        <div className="flashcard-actions">
          <button type="button" className="flashcard-next-btn" onClick={advance}>
            Still learning →
          </button>
          <button type="button" className="flashcard-mastered-btn" onClick={markMastered}>
            Mastered ✓
          </button>
        </div>
      )}
    </div>
  )
}

export default FlashcardRun
