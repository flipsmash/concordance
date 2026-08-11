import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import './SuggestWord.css'

const API_BASE = ''

// Admin-only "add a word directly" flow: search every dictionary source
// independently for a lemma, show each one's own answer side by side, let
// the admin pick one (or write from scratch) and edit it, then record it.
// The word is inserted book-less (no word_book row) -- a future book using
// it attaches automatically via sync_book_results' own upsert, and it's
// already judge-exempt for every future book the moment active=true is set
// (fetch_known_verdicts treats any active word as a cached "keep"), so
// neither of those needs anything built here.
function SuggestWord() {
  const navigate = useNavigate()
  const [lemma, setLemma] = useState('')
  const [posOptions, setPosOptions] = useState([])
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [result, setResult] = useState(null)
  const [form, setForm] = useState(null)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/api/pos-values`)
      .then((res) => res.json())
      .then(setPosOptions)
      .catch(() => {})
  }, [])

  function handleSearch(e) {
    e.preventDefault()
    const trimmed = lemma.trim()
    if (!trimmed) return
    setSearching(true)
    setSearchError('')
    setResult(null)
    setForm(null)
    setSaveError('')
    fetch(`${API_BASE}/api/admin/suggest-word/search?${new URLSearchParams({ lemma: trimmed })}`)
      .then((res) => (res.ok ? res.json() : res.json().then((body) => Promise.reject(new Error(body.detail || 'search failed')))))
      .then(setResult)
      .catch((err) => setSearchError(err.message))
      .finally(() => setSearching(false))
  }

  function selectCandidate(candidate) {
    setForm({
      lemma: result.lemma,
      definition: candidate.definition,
      part_of_speech: candidate.part_of_speech,
      ipa: candidate.ipa,
      etymology: candidate.etymology,
      synonyms: candidate.synonyms.join(', '),
      definition_source: candidate.source,
    })
  }

  function writeFromScratch() {
    setForm({
      lemma: result.lemma,
      definition: '',
      part_of_speech: '',
      ipa: '',
      etymology: '',
      synonyms: '',
      definition_source: 'admin (hand-written)',
    })
  }

  function updateForm(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  function handleSave(e) {
    e.preventDefault()
    setSaving(true)
    setSaveError('')
    fetch(`${API_BASE}/api/admin/suggest-word/finalize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lemma: form.lemma,
        definition: form.definition,
        part_of_speech: form.part_of_speech,
        ipa: form.ipa,
        etymology: form.etymology,
        synonyms: form.synonyms.split(',').map((s) => s.trim()).filter(Boolean),
        definition_source: form.definition_source,
      }),
    })
      .then((res) => (res.ok ? res.json() : res.json().then((body) => Promise.reject(new Error(body.detail || 'save failed')))))
      .then((word) => navigate(`/app/words/${word.id}`))
      .catch((err) => {
        setSaveError(err.message)
        setSaving(false)
      })
  }

  const posSelectOptions =
    form && form.part_of_speech && !posOptions.includes(form.part_of_speech)
      ? [...posOptions, form.part_of_speech]
      : posOptions

  return (
    <div className="suggest-word">
      <form className="suggest-word-search" onSubmit={handleSearch}>
        <label>
          Word:{' '}
          <input
            type="text"
            value={lemma}
            onChange={(e) => setLemma(e.target.value)}
            placeholder="e.g. perendinate"
            disabled={searching}
          />
        </label>
        <button type="submit" className="accept-btn" disabled={searching || !lemma.trim()}>
          {searching ? 'Searching…' : 'Search'}
        </button>
      </form>
      {searching && (
        <p className="suggest-word-hint">
          Checking every dictionary source, including a web-search fallback if nothing else has it — this can take
          up to ~15s.
        </p>
      )}
      {searchError && <div className="error-banner">{searchError}</div>}

      {result && result.exists && (
        <div className="suggest-word-exists">
          <p>
            <strong>{result.lemma}</strong> is already in the collection
            {result.active ? '' : ' (currently pruned/inactive)'}.
          </p>
          <Link to={`/app/words/${result.word_id}`}>View it</Link>
        </div>
      )}

      {result && !result.exists && !form && (
        <div className="suggest-word-candidates">
          {result.web_search_unavailable && (
            <p className="suggest-word-hint">
              Web-search fallback unavailable right now (the local model is likely in use by another job) — showing
              results from every other source.
            </p>
          )}
          {result.candidates.length === 0 && <p>No source had a definition for "{result.lemma}".</p>}
          {result.candidates.map((c, i) => (
            <div className="suggest-word-card" key={i}>
              <div className="suggest-word-card-header">
                <span className="suggest-word-card-source">{c.source}</span>
                {c.part_of_speech && <span className="suggest-word-card-pos">{c.part_of_speech}</span>}
              </div>
              <p className="suggest-word-card-definition">{c.definition || <em>no definition</em>}</p>
              {c.ipa && <p className="suggest-word-card-meta">IPA: {c.ipa}</p>}
              {c.etymology && <p className="suggest-word-card-meta">Etymology: {c.etymology}</p>}
              {c.synonyms.length > 0 && <p className="suggest-word-card-meta">Synonyms: {c.synonyms.join(', ')}</p>}
              {c.junk_pos_warning && (
                <p className="suggest-word-card-warning">
                  Dictionary resolves this as {c.part_of_speech} ({c.junk_pos_warning}) — normally excluded, your
                  call is final if you use it.
                </p>
              )}
              <button type="button" className="accept-btn" onClick={() => selectCandidate(c)}>
                Use this
              </button>
            </div>
          ))}
          <button type="button" className="suggest-word-scratch-btn" onClick={writeFromScratch}>
            Write from scratch
          </button>
        </div>
      )}

      {form && (
        <form className="suggest-word-edit" onSubmit={handleSave}>
          <h3>Add "{form.lemma}"</h3>
          <label>
            Definition
            <textarea
              value={form.definition}
              onChange={(e) => updateForm('definition', e.target.value)}
              rows={3}
              required
            />
          </label>
          <label>
            Part of speech
            <select value={form.part_of_speech} onChange={(e) => updateForm('part_of_speech', e.target.value)}>
              <option value="">—</option>
              {posSelectOptions.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <label>
            IPA
            <input type="text" value={form.ipa} onChange={(e) => updateForm('ipa', e.target.value)} />
          </label>
          <label>
            Etymology
            <input type="text" value={form.etymology} onChange={(e) => updateForm('etymology', e.target.value)} />
          </label>
          <label>
            Synonyms (comma-separated)
            <input type="text" value={form.synonyms} onChange={(e) => updateForm('synonyms', e.target.value)} />
          </label>
          <label>
            Definition source
            <input
              type="text"
              value={form.definition_source}
              onChange={(e) => updateForm('definition_source', e.target.value)}
            />
          </label>
          {saveError && <div className="error-banner">{saveError}</div>}
          <div className="suggest-word-edit-actions">
            <button type="button" onClick={() => setForm(null)} disabled={saving}>
              Back
            </button>
            <button type="submit" className="accept-btn" disabled={saving || !form.definition.trim()}>
              {saving ? 'Saving…' : 'Save word'}
            </button>
          </div>
        </form>
      )}
    </div>
  )
}

export default SuggestWord
