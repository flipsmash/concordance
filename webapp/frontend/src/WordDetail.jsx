import { useEffect, useState } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import GraphView from './GraphView'
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

function WordDetail({ backTo = '/accepted' }) {
  const { id } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const [word, setWord] = useState(null)
  const [notFound, setNotFound] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [neighbors, setNeighbors] = useState(null) // null = not loaded yet, [] = loaded, none found
  const [surpriseLoading, setSurpriseLoading] = useState(false)
  const [booksExpanded, setBooksExpanded] = useState(false)

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

  useEffect(() => {
    setLoading(true)
    setNotFound(false)
    setError('')
    setWord(null)
    setNeighbors(null)
    setBooksExpanded(false)

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
        <Link to={backTo}>← Back</Link>
      </div>
    )
  }

  if (error || !word) {
    return (
      <div className="word-detail">
        <div className="error-banner">{error || 'failed to load word'}</div>
        <Link to={backTo}>← Back</Link>
      </div>
    )
  }

  const hasAudio = word.audio_source && word.audio_source !== 'none'
  const factors = word.difficulty_factors

  return (
    <div className="word-detail">
      <Link to={backTo} className="word-detail-back">
        ← Back
      </Link>

      <div className="word-detail-header">
        <h1>{word.lemma}</h1>
        {word.part_of_speech && <span className="word-detail-pos">{word.part_of_speech}</span>}
        {hasAudio && (
          <audio controls src={`${API_BASE}/api/words/${id}/audio`} className="word-detail-audio" />
        )}
        <button
          type="button"
          className="word-detail-surprise"
          onClick={surpriseMe}
          disabled={surpriseLoading}
        >
          {surpriseLoading ? '…' : '🎲 Surprise me'}
        </button>
      </div>

      <section className="word-detail-section">
        <p className="word-detail-definition">{word.definition || '—'}</p>
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

      <section className="word-detail-section">
        <h2>Source</h2>
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
