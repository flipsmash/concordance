import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import NgramTrendChart from './NgramTrendChart'
import './Popularity.css'

const API_BASE = ''
const MAX_TERMS = 4
const EXAMPLES = [
  ['telegraph', 'wireless', 'radio'],
  ['steed', 'horse'],
  ['thou', 'you'],
  ['groovy', 'awesome'],
]

function parseTerms(raw) {
  const seen = []
  for (const t of (raw || '').split(',')) {
    const term = t.trim().toLowerCase()
    if (term && !seen.includes(term)) seen.push(term)
  }
  return seen.slice(0, MAX_TERMS)
}

// Popularity of one to four terms over time in Google Books, from the local
// per-decade Ngram table. Terms live in the URL (?terms=a,b) so a
// comparison is linkable -- the word pages and the Visualizations hub
// deep-link here.
function Popularity() {
  const [searchParams, setSearchParams] = useSearchParams()
  const terms = parseTerms(searchParams.get('terms'))
  const termsKey = terms.join(',')
  const [input, setInput] = useState(terms.join(', '))
  const [logScale, setLogScale] = useState(false)
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setInput(termsKey.split(',').filter(Boolean).join(', '))
    if (!termsKey) { setData(null); return }
    setError('')
    const qs = new URLSearchParams(termsKey.split(',').map((t) => ['term', t]))
    fetch(`${API_BASE}/api/browse/ngram-trend?${qs}`)
      .then((r) => (r.ok ? r.json() : r.json().then((b) => Promise.reject(new Error(b.detail || r.status)))))
      .then(setData)
      .catch((err) => setError(err.message || 'failed to load'))
  }, [termsKey])

  function show(list) {
    const next = parseTerms(list.join(','))
    setSearchParams(next.length ? { terms: next.join(',') } : {})
  }

  const missing = data ? data.series.filter((s) => !s.found).map((s) => s.term) : []
  const found = data ? data.series.filter((s) => s.found) : []

  return (
    <div className="pop-page">
      <Link to="/app/visualizations" className="pop-crumb">← Visualizations</Link>
      <h1 className="pop-title">Popularity over time</h1>
      <p className="pop-lede">
        How often a word appeared in English books, decade by decade from the 1800s to the 2010s,
        in occurrences per million words. Compare up to {MAX_TERMS}.
      </p>

      <form
        className="pop-form"
        onSubmit={(e) => { e.preventDefault(); show(input.split(',')) }}
      >
        <input
          className="pop-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. telegraph, wireless, radio"
          aria-label="Terms to compare, separated by commas"
        />
        <button type="submit" className="pop-button">Show</button>
        <label className="pop-log">
          <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} />
          Log scale
        </label>
      </form>

      <div className="pop-examples">
        Try:{' '}
        {EXAMPLES.map((ex) => (
          <button key={ex.join()} type="button" className="pop-example" onClick={() => show(ex)}>
            {ex.join(' · ')}
          </button>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {found.length > 1 && (
        <div className="pop-legend trend-root">
          {found.map((s) => (
            <span key={s.term} className="pop-chip">
              <span className={`trend-swatch trend-slot-${data.series.indexOf(s) + 1}`} />
              {s.term}
              <button
                type="button"
                className="pop-chip-x"
                aria-label={`Remove ${s.term}`}
                onClick={() => show(terms.filter((t) => t !== s.term))}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      {missing.length > 0 && (
        <p className="pop-missing">
          Not in Google Books (lowercase, as a single word): {missing.join(', ')}
        </p>
      )}

      {found.length > 0 && <NgramTrendChart data={data} logScale={logScale} />}

      <p className="pop-source">
        Source: Google Books Ngram, English 2019 corpus, lowercase exact match, per decade
        (the 2010s cover 2010–2019).
      </p>
    </div>
  )
}

export default Popularity
