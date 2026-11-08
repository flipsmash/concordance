import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import OriginSunburst from './OriginSunburst'
import OriginTreemap from './OriginTreemap'
import { FAMILY_ORDER, ROUTE_MIX, familyColor, routeLabel } from './originColors'
import './GraphView.css' // .graph-maximize / .graph-signal-toggle
import { VizPageHeader } from './VizPage'
import './Origins.css'

const API_BASE = ''
const TABS = [
  { id: 'sunburst', label: 'Sunburst' },
  { id: 'treemap', label: 'Treemap' },
]
const PAGE = 60

function useOrigins(genre) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    if (genre === null) {        // the compare slot, unused: no request
      setData(null)
      return undefined
    }
    let live = true
    setError('')
    fetch(`${API_BASE}/api/browse/origins${genre ? `?genre=${encodeURIComponent(genre)}` : ''}`)
      .then((r) => {
        if (!r.ok) throw new Error(`request failed (${r.status})`)
        return r.json()
      })
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e.message))
    return () => {
      live = false
    }
  }, [genre])
  return { data, error }
}

const scopeName = (g) => g || 'All books'

// Where the collection's words come from: each word's earliest attested source
// language (root family) and how it reached English. Sunburst = family >
// route; treemap = family > language. Optional genre filter and a second,
// side-by-side scope to compare (both in percentages, so a small genre and
// the whole corpus are comparable).
function Origins() {
  const [params, setParams] = useSearchParams()
  const tab = TABS.some((t) => t.id === params.get('tab')) ? params.get('tab') : 'sunburst'
  const genre = params.get('genre') || ''
  const comparing = params.has('vs')
  const vs = params.get('vs') || ''
  const [genres, setGenres] = useState([])
  const [selected, setSelected] = useState(null) // {family, route?, lang?, langName?, genre}
  const [words, setWords] = useState(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const rootRef = useRef(null)
  const a = useOrigins(genre)
  const b = useOrigins(comparing ? vs : null)

  function update(changes) {
    const next = new URLSearchParams(params)
    for (const [k, v] of Object.entries(changes)) {
      if (v === null || v === undefined) next.delete(k)
      else next.set(k, v)
    }
    setParams(next, { replace: true })
  }

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/genres`)
      .then((r) => (r.ok ? r.json() : []))
      .then(setGenres)
      .catch(() => setGenres([]))
  }, [])

  useEffect(() => {
    const onChange = () => setIsFullscreen(document.fullscreenElement === rootRef.current)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  useEffect(() => setSelected(null), [genre, vs, tab])

  // The words behind the selected segment.
  useEffect(() => {
    if (!selected) {
      setWords(null)
      return
    }
    let live = true
    setWords(null)
    const q = new URLSearchParams({ family: selected.family, limit: String(PAGE) })
    if (selected.route) q.set('route', selected.route)
    if (selected.lang) q.set('lang', selected.lang)
    if (selected.genre) q.set('genre', selected.genre)
    fetch(`${API_BASE}/api/browse/origins/words?${q}`)
      .then((r) => (r.ok ? r.json() : { total: 0, words: [] }))
      .then((d) => live && setWords(d))
      .catch(() => live && setWords({ total: 0, words: [] }))
    return () => {
      live = false
    }
  }, [selected])

  function loadMore() {
    const q = new URLSearchParams({ family: selected.family, limit: String(PAGE), offset: String(words.words.length) })
    if (selected.route) q.set('route', selected.route)
    if (selected.lang) q.set('lang', selected.lang)
    if (selected.genre) q.set('genre', selected.genre)
    fetch(`${API_BASE}/api/browse/origins/words?${q}`)
      .then((r) => r.json())
      .then((d) => setWords((w) => ({ total: w.total, words: [...w.words, ...d.words] })))
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen()
    else rootRef.current?.requestFullscreen()
  }

  const Chart = tab === 'treemap' ? OriginTreemap : OriginSunburst
  const scopes = [{ key: 'a', genre, q: a }, ...(comparing ? [{ key: 'b', genre: vs, q: b }] : [])]
  const ready = scopes.every((s) => s.q.data)
  const error = scopes.map((s) => s.q.error).find(Boolean)

  const present = FAMILY_ORDER.filter((f) => scopes.some((s) => s.q.data?.families.some((x) => x.family === f && x.count)))
  const share = (d, fam) => {
    const f = d?.families.find((x) => x.family === fam)
    return f && d.known_words ? (f.count / d.known_words) * 100 : 0
  }
  const count = (d, fam) => d?.families.find((x) => x.family === fam)?.count || 0

  return (
    <div className="origins-page" ref={rootRef}>
      <VizPageHeader title="Where the words come from">
        Each word&rsquo;s earliest recorded source language and how it reached English, from Wiktionary,
        the 0 Dict and word formation.
      </VizPageHeader>

      <div className="origins-filters">
        <label className="origins-filter">
          <span className="muted">Books</span>
          <select value={genre} onChange={(e) => update({ genre: e.target.value || null })}>
            <option value="">All books</option>
            {genres.map((g) => <option key={g} value={g}>{g}</option>)}
          </select>
        </label>
        <label className="origins-filter">
          <span className="muted">Compare with</span>
          <select value={comparing ? vs || '__all__' : ''}
            onChange={(e) => update({ vs: e.target.value === '' ? null : e.target.value === '__all__' ? '' : e.target.value })}>
            <option value="">— nothing —</option>
            <option value="__all__">All books</option>
            {genres.map((g) => <option key={g} value={g}>{g}</option>)}
          </select>
        </label>
        <div className="graph-signal-toggle" role="group" aria-label="View">
          {TABS.map((t) => (
            <button key={t.id} type="button" className={tab === t.id ? 'active' : ''}
              onClick={() => update({ tab: t.id === 'sunburst' ? null : t.id })}>
              {t.label}
            </button>
          ))}
        </div>
        <button type="button" className="graph-maximize origins-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {!ready && !error && <p className="muted">Loading…</p>}

      {ready && (
        <>
          <p className="muted origins-coverage">
            {scopes.map((s, i) => (
              <span key={s.key}>
                {i > 0 && ' · '}
                {comparing && <strong>{scopeName(s.genre)}: </strong>}
                origins known for {s.q.data.known_words.toLocaleString()} of {s.q.data.total_words.toLocaleString()} words
                ({s.q.data.total_words ? Math.round((s.q.data.known_words / s.q.data.total_words) * 100) : 0}%)
              </span>
            ))}
            {' '}— the rest aren&rsquo;t in any dictionary source, and aren&rsquo;t guessed.
          </p>

          <div className={`origins-charts${comparing ? ' compare' : ''}`}>
            {scopes.map((s) => (
              <Chart
                key={s.key}
                families={s.q.data.families}
                known={s.q.data.known_words}
                label={comparing ? scopeName(s.genre) : null}
                selected={selected && selected.genre === s.genre ? selected : null}
                onSelect={(sel) => setSelected({ ...sel, genre: s.genre })}
              />
            ))}
          </div>

          <div className="origins-legend" aria-label="Legend">
            {present.map((f) => (
              <span key={f} className="origins-legend-item">
                <span className="origins-swatch" style={{ background: familyColor(f) }} />
                {f}
              </span>
            ))}
            {tab === 'sunburst' && (
              <span className="origins-legend-routes muted">
                Outer ring: full colour = borrowed directly (or inherited),{' '}
                {ROUTE_MIX.via}% paler = came via another language, {ROUTE_MIX.formed}% paler = formed in English.
              </span>
            )}
          </div>

          <div className="origins-lower">
            <table className="origins-table">
              <caption className="muted">
                {comparing ? 'Share of words with a known origin' : 'Words with a known origin, by root family'}. Click a row for its words.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Root family</th>
                  {comparing ? (
                    <>
                      <th scope="col" className="num">{scopeName(genre)}</th>
                      <th scope="col" className="num">{scopeName(vs)}</th>
                      <th scope="col" className="num">Difference</th>
                    </>
                  ) : (
                    <>
                      <th scope="col" className="num">Words</th>
                      <th scope="col" className="num">Share</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {present.map((f) => {
                  const pa = share(a.data, f)
                  const pb = share(b.data, f)
                  const sel = selected && selected.family === f && !selected.route && !selected.lang
                  return (
                    <tr key={f} className={sel ? 'selected' : ''} tabIndex={0}
                      onClick={() => setSelected({ family: f, genre })}
                      onKeyDown={(e) => e.key === 'Enter' && setSelected({ family: f, genre })}>
                      <th scope="row">
                        <span className="origins-swatch" style={{ background: familyColor(f) }} />
                        {f}
                      </th>
                      {comparing ? (
                        <>
                          <td className="num">{pa.toFixed(1)}%</td>
                          <td className="num">{pb.toFixed(1)}%</td>
                          <td className="num">{pa - pb >= 0 ? '+' : '−'}{Math.abs(pa - pb).toFixed(1)} pts</td>
                        </>
                      ) : (
                        <>
                          <td className="num">{count(a.data, f).toLocaleString()}</td>
                          <td className="num">{pa.toFixed(1)}%</td>
                        </>
                      )}
                    </tr>
                  )
                })}
              </tbody>
            </table>

            <section className="origins-words" aria-live="polite">
              {!selected && <p className="muted">Click any part of the chart, or a row of the table, to see its words.</p>}
              {selected && (
                <>
                  <h3>
                    <span className="origins-swatch" style={{ background: familyColor(selected.family) }} />
                    {selected.langName || selected.family}
                    {selected.route && ` › ${routeLabel(selected.route)}`}
                    {comparing && <span className="muted"> · {scopeName(selected.genre)}</span>}
                  </h3>
                  {words === null && <p className="muted">Loading…</p>}
                  {words && (
                    <>
                      <p className="muted">{words.total.toLocaleString()} words, the most widely read first</p>
                      <ul className="origins-word-list">
                        {words.words.map((w) => (
                          <li key={w.id}>
                            <Link to={`/app/words/${w.id}`}>{w.lemma}</Link>
                            <span className="muted"> — {w.language}{w.route !== 'direct' && w.route !== 'inherited' ? `, ${routeLabel(w.route)}` : ''}</span>
                            {w.definition && <span className="origins-def">{w.definition}</span>}
                          </li>
                        ))}
                      </ul>
                      {words.words.length < words.total && (
                        <button type="button" className="graph-maximize" onClick={loadMore}>Show more</button>
                      )}
                    </>
                  )}
                </>
              )}
            </section>
          </div>
        </>
      )}
    </div>
  )
}

export default Origins
