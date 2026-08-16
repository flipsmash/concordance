import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import OedMatchBadge from './OedMatchBadge'
import './Authors.css'
import './OedEntryDetail.css'

const API_BASE = ''

// Admin-only detail view for one oed.entry row (see webapp/backend/oed.py) --
// headword/pronunciation/etymology up top, then each definition with its
// quotations underneath, in the source's own sort_order.
function OedEntryDetail() {
  const { entryId } = useParams()
  const [entry, setEntry] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setEntry(null)
    setError('')
    fetch(`${API_BASE}/api/admin/oed/entries/${entryId}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to load entry'))))
      .then(setEntry)
      .catch((err) => setError(err.message))
  }, [entryId])

  if (error) return <div className="error-banner">{error}</div>
  if (!entry) return <div className="page-loading">Loading…</div>

  return (
    <div className="authors-page oed-entry-page">
      <header className="authors-header">
        <div>
          <h1>
            {entry.headword}
            {entry.homograph_number != null && <sup>{entry.homograph_number}</sup>}
          </h1>
          <Link to=".." relative="path" className="oed-back-link">← Back to entries</Link>
        </div>
      </header>

      <section className="oed-meta">
        {entry.part_of_speech && <span className="oed-meta-item">{entry.part_of_speech}</span>}
        <span className="oed-meta-item">page {entry.page_number}</span>
        <span className="oed-meta-item">{entry.entry_type}</span>
        {entry.pronunciation_ipa ? (
          <span className="oed-meta-item oed-ipa">/{entry.pronunciation_ipa}/</span>
        ) : (
          <span className="oed-meta-item oed-no-ipa">no pronunciation</span>
        )}
        {entry.pronunciation_needs_review && (
          <span className="oed-review-badge">needs review</span>
        )}
        <OedMatchBadge match={entry.concordance_match} />
        {entry.concordance_word_id != null && (
          <Link to={`/words/${entry.concordance_word_id}`} className="oed-back-link">
            View matching concordance word →
          </Link>
        )}
      </section>

      {entry.etymology && (
        <section className="oed-section">
          <h2 className="oed-section-heading">Etymology</h2>
          <p className="oed-etymology">[{entry.etymology}]</p>
        </section>
      )}

      {entry.pronunciation_raw && (
        <section className="oed-section">
          <h2 className="oed-section-heading">Raw OCR pronunciation</h2>
          <p className="oed-raw-pron">{entry.pronunciation_raw}</p>
        </section>
      )}

      <section className="oed-section">
        <h2 className="oed-section-heading">Definitions</h2>
        {entry.definitions.length === 0 && <p className="muted">No definitions parsed for this entry.</p>}
        <ol className="oed-definitions">
          {entry.definitions.map((d) => (
            <li key={d.id} className="oed-definition">
              {d.sense_label && <span className="oed-sense-label">{d.sense_label}.</span>}
              <span className="oed-definition-text">{d.definition_text}</span>
              {d.quotations.length > 0 && (
                <ul className="oed-quotations">
                  {d.quotations.map((q) => (
                    <li key={q.id} className="oed-quotation">
                      <span className="oed-quote-year">
                        {q.year_approx ? '~' : ''}{q.year ?? q.year_raw}
                      </span>
                      {q.author && <span className="oed-quote-author">{q.author}</span>}
                      <span className="oed-quote-text">{q.quoted_text}</span>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ol>
      </section>

      <section className="oed-section">
        <h2 className="oed-section-heading">Raw extracted text</h2>
        <p className="oed-raw-text">{entry.raw_text}</p>
      </section>
    </div>
  )
}

export default OedEntryDetail
