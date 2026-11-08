import { useState } from 'react'

const API_BASE = ''

// Admin-only: change a word's spelling (POST /api/words/{id}/spelling). The
// server renames it, keeps the old spelling as an alias, looks the
// definition up again and starts the rest of the upkeep in the background.
// When the new spelling already belongs to another word it answers 409 with
// that word, and this asks before merging into it.
function WordRespell({ word, onDone }) {
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [collision, setCollision] = useState(null)

  function start() {
    setDraft(word.lemma)
    setError('')
    setCollision(null)
    setOpen(true)
  }

  function close() {
    setOpen(false)
    setCollision(null)
    setError('')
  }

  function save(merge = false) {
    const spelling = draft.trim()
    if (!spelling || spelling === word.lemma) {
      close()
      return
    }
    setSaving(true)
    setError('')
    fetch(`${API_BASE}/api/words/${word.id}/spelling`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spelling, merge }),
    })
      .then(async (res) => {
        const body = await res.json().catch(() => ({}))
        if (res.status === 409) {
          setCollision(body.detail.collision)
          return
        }
        if (!res.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `failed (${res.status})`)
        close()
        onDone(body)
      })
      .catch((err) => setError(err.message || 'failed to change the spelling'))
      .finally(() => setSaving(false))
  }

  if (!open) {
    return (
      <button type="button" className="word-detail-edit-definition" onClick={start} title="Change this word's spelling">
        ✎ Spelling
      </button>
    )
  }

  return (
    <div className="word-respell">
      <form className="word-respell-form" onSubmit={(e) => { e.preventDefault(); save(false) }}>
        <label className="muted" htmlFor="word-respell-input">Spelling</label>
        <input
          id="word-respell-input"
          value={draft}
          onChange={(e) => { setDraft(e.target.value); setCollision(null) }}
          disabled={saving}
          autoFocus
        />
        <button type="button" onClick={close} disabled={saving}>Cancel</button>
        <button type="submit" className="accept-btn" disabled={saving || !!collision}>
          {saving ? 'Saving…' : 'Save'}
        </button>
      </form>
      <p className="muted word-respell-hint">
        “{word.lemma}” stays as an alias, so books that use it still attach here. The definition is looked up
        again if this word has none; categories, difficulty and the rest update in the background.
      </p>
      {collision && (
        <div className="word-respell-merge">
          <p>
            <strong>“{collision.lemma}” is already a word</strong>
            {collision.books > 0 && <> (in {collision.books.toLocaleString()} {collision.books === 1 ? 'book' : 'books'})</>}.
            Merge “{word.lemma}” into it? Its books, quiz history, sets and progress move to “{collision.lemma}”, and
            “{word.lemma}” becomes an alias.
          </p>
          {!collision.active && (
            <p>
              “{collision.lemma}” was removed earlier ({collision.inactive_reason.replaceAll('_', ' ')}); merging
              brings it back.
            </p>
          )}
          <div className="word-detail-definition-edit-actions">
            <button type="button" onClick={() => setCollision(null)} disabled={saving}>Cancel</button>
            <button type="button" className="accept-btn" onClick={() => save(true)} disabled={saving}>
              {saving ? 'Merging…' : `Merge into “${collision.lemma}”`}
            </button>
          </div>
        </div>
      )}
      {error && <div className="error-banner">{error}</div>}
    </div>
  )
}

export default WordRespell
