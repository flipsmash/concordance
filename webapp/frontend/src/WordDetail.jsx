import { useEffect, useState } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import AddToSetMenu from './AddToSetMenu'
import { useAuth } from './AuthContext'
import GraphView from './GraphView'
import LinkedDefinition from './LinkedDefinition'
import { colorForBucket } from './domainColors'
import './WordDetail.css'

const API_BASE = ''
const BOOKS_PREVIEW_COUNT = 10

function CategoryChip({ category }) {
  return (
    <span
      className={category.is_primary ? 'category-chip primary' : 'category-chip'}
      style={{ background: colorForBucket(category.color_bucket) }}
      title={category.confidence != null ? `confidence ${Math.round(category.confidence * 100)}%` : undefined}
    >
      {category.domain_name ? `${category.domain_name} - ${category.name}` : category.name}
    </span>
  )
}

function WordDetail({ backTo = '/app/admin/accepted', showBackLink = true }) {
  const { id } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const { user } = useAuth()
  const [word, setWord] = useState(null)
  const [notFound, setNotFound] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(false)
  const [neighbors, setNeighbors] = useState(null) // null = not loaded yet, [] = loaded, none found
  const [surpriseLoading, setSurpriseLoading] = useState(false)
  const [booksExpanded, setBooksExpanded] = useState(false)
  const [progressHistory, setProgressHistory] = useState(null) // null = not loaded / unavailable, object = loaded
  const [editingDefinition, setEditingDefinition] = useState(false)
  const [definitionDraft, setDefinitionDraft] = useState('')
  const [savingDefinition, setSavingDefinition] = useState(false)
  const [definitionError, setDefinitionError] = useState('')

  // Same host-route family as this page's own -- /app/words/:id keeps
  // navigating within /app (so `backTo="/app"` keeps working after repeat
  // clicks), /words/:id (the admin curation detail view) stays on /words.
  const wordsBase = location.pathname.startsWith('/app/') ? '/app/words' : '/words'

  function surpriseMe() {
    setSurpriseLoading(true)
    fetch(`${API_BASE}/api/browse/words?random=true`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        const next = data?.items?.[0]
        if (next) navigate(`${wordsBase}/${next.id}`)
      })
      .catch(() => {})
      .finally(() => setSurpriseLoading(false))
  }

  // Admin-only. Same soft-delete endpoint (DELETE /api/words/{id}) the admin
  // curation view's AcceptedView.jsx already uses -- word.active becomes
  // false, so it drops out of every downstream view rather than being
  // hard-deleted. Navigates back on success since there's nothing left here
  // to show.
  function deleteWord() {
    if (!window.confirm(`Delete "${word.lemma}"? This removes it from the active vocabulary.`)) return
    setDeleting(true)
    fetch(`${API_BASE}/api/words/${id}`, { method: 'DELETE' })
      .then((res) => {
        if (!res.ok) throw new Error(`failed to delete (${res.status})`)
        navigate(backTo)
      })
      .catch((err) => {
        setError(err.message || 'failed to delete word')
        setDeleting(false)
      })
  }

  // Admin-only. Server notes the edit (previous value + who/when) but
  // nothing about that is surfaced here -- the definition just updates in
  // place like any other field, same as every other admin action on this
  // page (delete, prune) leaves no "edited" marker of its own either.
  function startEditingDefinition() {
    setDefinitionDraft(word.definition || '')
    setDefinitionError('')
    setEditingDefinition(true)
  }

  function cancelEditingDefinition() {
    setEditingDefinition(false)
    setDefinitionError('')
  }

  function saveDefinition() {
    const trimmed = definitionDraft.trim()
    if (!trimmed) {
      setDefinitionError('definition cannot be empty')
      return
    }
    setSavingDefinition(true)
    setDefinitionError('')
    fetch(`${API_BASE}/api/words/${id}/definition`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ definition: trimmed }),
    })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`failed to save (${res.status})`))))
      .then((data) => {
        // definition_links cleared client-side too -- the backend just
        // deleted this word's stale word_definition_link rows (computed
        // against the OLD text; matching them against the new text risks a
        // coincidental-substring link to the wrong word). Recomputed fresh
        // on the next `concordance link-definitions` run.
        setWord((prev) => ({ ...prev, definition: data.definition, definition_links: [] }))
        setEditingDefinition(false)
      })
      .catch((err) => setDefinitionError(err.message || 'failed to save definition'))
      .finally(() => setSavingDefinition(false))
  }

  useEffect(() => {
    setLoading(true)
    setNotFound(false)
    setError('')
    setWord(null)
    setNeighbors(null)
    setBooksExpanded(false)
    setProgressHistory(null)

    // Separate, best-effort fetch -- this is require_user-gated personal data
    // (unlike /api/words/{id}, which works for the account-less viewer path
    // too), so a 401 here (viewer not logged in) or a 404 is expected and
    // must render nothing, never an error banner, since WordDetail also
    // mounts on the account-less admin/curation route.
    fetch(`${API_BASE}/api/progress/words/${id}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setProgressHistory(data))
      .catch(() => setProgressHistory(null))

    fetch(`${API_BASE}/api/words/${id}`)
      .then((res) => {
        if (res.status === 404) {
          setNotFound(true)
          return null
        }
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => data && setWord(data))
      .catch((err) => setError(err.message || 'failed to load word'))
      .finally(() => setLoading(false))

    fetch(`${API_BASE}/api/words/${id}/neighbors?signal=definition&k=8`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setNeighbors(data?.neighbors ?? []))
      .catch(() => setNeighbors([]))
  }, [id])

  if (loading) return <div className="word-detail-loading">Loading…</div>

  if (notFound) {
    return (
      <div className="word-detail word-detail-not-found">
        <p>Word not found.</p>
        {showBackLink && <Link to={backTo}>← Back</Link>}
      </div>
    )
  }

  if (error || !word) {
    return (
      <div className="word-detail">
        <div className="error-banner">{error || 'failed to load word'}</div>
        {showBackLink && <Link to={backTo}>← Back</Link>}
      </div>
    )
  }

  const hasAudio = word.audio_source && word.audio_source !== 'none'
  const factors = word.difficulty_factors

  return (
    <div className="word-detail">
      {showBackLink && (
        <Link to={backTo} className="word-detail-back">
          ← Back
        </Link>
      )}

      <div className="word-detail-header">
        <h1>{word.lemma}</h1>
        {word.part_of_speech && <span className="word-detail-pos">{word.part_of_speech}</span>}
        {hasAudio && (
          <audio controls src={`${API_BASE}/api/words/${id}/audio`} className="word-detail-audio" />
        )}
        <AddToSetMenu wordIds={[word.id]} />
        <button
          type="button"
          className="word-detail-surprise"
          onClick={surpriseMe}
          disabled={surpriseLoading}
        >
          {surpriseLoading ? '…' : '🎲 Surprise me'}
        </button>
        {user?.is_admin && (
          <button
            type="button"
            className="word-detail-delete"
            onClick={deleteWord}
            disabled={deleting}
          >
            {deleting ? '…' : '🗑 Delete'}
          </button>
        )}
      </div>

      <section className="word-detail-section">
        {editingDefinition ? (
          <div className="word-detail-definition-edit">
            <textarea
              value={definitionDraft}
              onChange={(e) => setDefinitionDraft(e.target.value)}
              rows={3}
              disabled={savingDefinition}
              autoFocus
            />
            {definitionError && <div className="error-banner">{definitionError}</div>}
            <div className="word-detail-definition-edit-actions">
              <button type="button" onClick={cancelEditingDefinition} disabled={savingDefinition}>
                Cancel
              </button>
              <button type="button" className="accept-btn" onClick={saveDefinition} disabled={savingDefinition}>
                {savingDefinition ? 'Saving…' : 'Save'}
              </button>
            </div>
          </div>
        ) : (
          <p className="word-detail-definition">
            {word.definition ? <LinkedDefinition text={word.definition} links={word.definition_links} /> : '—'}
            {user?.is_admin && (
              <button type="button" className="word-detail-edit-definition" onClick={startEditingDefinition}>
                ✎ Edit
              </button>
            )}
          </p>
        )}
        {word.etymology && <p className="word-detail-etymology">{word.etymology}</p>}
        {word.synonyms.length > 0 && (
          <div className="word-detail-synonyms">
            {word.synonyms.map((s) => (
              <span className="synonym-chip" key={s}>
                {s}
              </span>
            ))}
          </div>
        )}
      </section>

      {word.sentence && (
        <section className="word-detail-section">
          <blockquote className="word-detail-sentence">
            {word.sentence}
            {word.chapter && <cite> — {word.chapter}</cite>}
          </blockquote>
        </section>
      )}

      {word.ipa && (
        <section className="word-detail-section">
          <h2>Pronunciation</h2>
          <span className="word-detail-ipa">/{word.ipa}/</span>
        </section>
      )}

      <section className="word-detail-section">
        <h2>Categorization</h2>
        {word.categories.length > 0 ? (
          <div className="word-detail-categories">
            {word.categories.map((c) => (
              <CategoryChip category={c} key={c.code} />
            ))}
          </div>
        ) : (
          <p className="muted">Not yet categorized.</p>
        )}
      </section>

      <section className="word-detail-section difficulty-panel">
        <h2>Difficulty</h2>
        <div className="difficulty-score">
          {word.difficulty != null ? Math.round(word.difficulty) : '—'}
          <span className="difficulty-scale">/100</span>
        </div>
        {factors && <p className="difficulty-why">{factors.why}</p>}
        <div className="word-detail-badges">
          {word.archaic && word.archaic !== 'current' && (
            <span className="register-badge">
              {word.archaic}
              {word.archaic_confidence != null && ` (${Math.round(word.archaic_confidence * 100)}%)`}
            </span>
          )}
          {word.quizzable != null && (
            <span className={word.quizzable ? 'quizzable-badge' : 'quizzable-badge not-quizzable'}>
              {word.quizzable ? 'Quizzable' : `Not quizzable${word.quizzable_reason ? `: ${word.quizzable_reason}` : ''}`}
            </span>
          )}
        </div>
        {/* Admin-only: the quizzable verdict above is judged against
            quiz_definition (what a quiz actually shows), NOT the plain
            `definition` displayed at the top of this page -- those two can
            read very differently (a leaking raw definition, safely
            rewritten for quizzing) with nothing else on this page to make
            that visible. Found live: an admin flagged "codpieced" as a bug
            because its definition still says "codpiece", with no way to
            see that quiz_definition had already been rewritten leak-free. */}
        {user?.is_admin && word.quiz_definition && (
          <p className="muted word-detail-quiz-definition">
            Quiz definition ({word.quiz_def_source}): {word.quiz_definition}
          </p>
        )}
        {factors && (
          <div className="difficulty-factors-grid">
            <span>zipf</span>
            <span>{factors.zipf.toFixed(2)}</span>
            <span>rarity</span>
            <span>{factors.rarity.toFixed(3)}</span>
            <span>archaic</span>
            <span>{factors.archaic.toFixed(3)}</span>
            <span>domain</span>
            <span>{factors.domain.toFixed(3)}</span>
            <span>morph</span>
            <span>{factors.morph.toFixed(3)}</span>
          </div>
        )}
        <div className="difficulty-factors-grid">
          <span>zipf (live)</span>
          <span>{word.zipf.toFixed(2)}</span>
          {word.ngram_peak != null && (
            <>
              <span>ngram peak</span>
              <span>{word.ngram_peak.toExponential(2)}</span>
            </>
          )}
          {word.ngram_recent != null && (
            <>
              <span>ngram recent</span>
              <span>{word.ngram_recent.toExponential(2)}</span>
            </>
          )}
          {word.ngram_recency_ratio != null && (
            <>
              <span>recency ratio</span>
              <span>{word.ngram_recency_ratio.toFixed(2)}</span>
            </>
          )}
          {word.ngram_peak_year != null && (
            <>
              <span>peak year</span>
              <span>{word.ngram_peak_year}</span>
            </>
          )}
        </div>
      </section>

      {progressHistory && progressHistory.answers.length > 0 && (
        <section className="word-detail-section your-history-panel">
          <h2>Your history with this word</h2>
          <div className="your-history-dots">
            {progressHistory.answers.slice(-20).map((a, i) => (
              <span
                key={i}
                className={a.is_correct ? 'quiz-review-mark correct' : 'quiz-review-mark incorrect'}
                title={new Date(a.answered_at).toLocaleDateString()}
              >
                {a.is_correct ? '✓' : '✕'}
              </span>
            ))}
            {progressHistory.answers.length > 20 && (
              <span className="your-history-more">+{progressHistory.answers.length - 20} more</span>
            )}
          </div>
          <div className="difficulty-factors-grid">
            <span>Streak</span>
            <span>{progressHistory.streak}</span>
            <span>Correct / incorrect</span>
            <span>{progressHistory.correct_count} / {progressHistory.incorrect_count}</span>
            {progressHistory.next_eligible_at && (
              <>
                <span>Next review</span>
                <span>{new Date(progressHistory.next_eligible_at).toLocaleDateString()}</span>
              </>
            )}
            {progressHistory.personal_difficulty != null && (
              <>
                <span>Your difficulty</span>
                <span>
                  {Math.round(progressHistory.personal_difficulty)}/100
                  {word.difficulty != null &&
                    ` (population: ${Math.round(word.difficulty)}/100)`}
                </span>
              </>
            )}
          </div>
        </section>
      )}
      {progressHistory && progressHistory.answers.length === 0 && (
        <section className="word-detail-section your-history-panel">
          <h2>Your history with this word</h2>
          <p className="muted">You haven't been quizzed on this word yet.</p>
        </section>
      )}

      <section className="word-detail-section">
        <h2>
          {word.books.length === 1 ? 'Source' : 'Sources'}
          {word.books.length > 0 && ` (${word.books.length})`}
        </h2>
        {word.books.length > 0 ? (
          <>
            <ul className="source-books-list">
              {(booksExpanded ? word.books : word.books.slice(0, BOOKS_PREVIEW_COUNT)).map((b) => (
                <li key={b.id}>
                  {b.author ? (
                    <Link to={`/app/authors/${encodeURIComponent(b.author)}/${b.id}`}>{b.title}</Link>
                  ) : (
                    b.title
                  )}
                </li>
              ))}
            </ul>
            {word.books.length > BOOKS_PREVIEW_COUNT && (
              <button
                type="button"
                className="word-detail-books-toggle"
                onClick={() => setBooksExpanded((v) => !v)}
              >
                {booksExpanded ? 'Show fewer' : `Show all ${word.books.length}`}
              </button>
            )}
          </>
        ) : (
          <p>—</p>
        )}
      </section>

      <section className="word-detail-section">
        <h2>Similar words</h2>
        {neighbors === null ? (
          <p className="muted">Loading…</p>
        ) : neighbors.length > 0 ? (
          <ul className="similar-words-list">
            {neighbors.map((n) => (
              <li key={n.id}>
                <Link to={`/words/${n.id}`}>{n.lemma}</Link>
                {n.definition && <span className="similar-word-definition"> — {n.definition}</span>}
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">No similar words yet — this word hasn't been embedded.</p>
        )}
      </section>

      <section className="word-detail-section">
        <h2>Graph</h2>
        <div className="word-detail-graph">
          <GraphView
            initialWordId={word.id}
            onNodeNavigate={(node) => navigate(`/words/${node.id}`)}
            hideSearch
          />
        </div>
      </section>
    </div>
  )
}

export default WordDetail
