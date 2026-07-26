import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import './Progress.css'
import SingleHueBarChart from './SingleHueBarChart'
import SparkTrendLine from './SparkTrendLine'

const API_BASE = ''

function useSortableRows(rows, defaultSort) {
  const [sort, setSort] = useState(defaultSort)
  const [dir, setDir] = useState('desc')

  function toggleSort(key) {
    if (sort === key) {
      setDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSort(key)
      setDir('desc')
    }
  }

  const sorted = rows ? [...rows].sort((a, b) => {
    const av = a[sort]
    const bv = b[sort]
    if (av === bv) return 0
    const cmp = av == null ? -1 : bv == null ? 1 : av < bv ? -1 : 1
    return dir === 'asc' ? cmp : -cmp
  }) : []

  return { sorted, sort, dir, toggleSort }
}

function SortHeader({ label, sortKey, sort, dir, onSort }) {
  const active = sort === sortKey
  return (
    <th onClick={() => onSort(sortKey)} className={active ? 'progress-th-active' : ''}>
      {label}{active ? (dir === 'asc' ? ' ▲' : ' ▼') : ''}
    </th>
  )
}

function StatTile({ label, value, unit, n }) {
  const display = value === null || value === undefined ? '—' : `${value}${unit ?? ''}`
  return (
    <div className="progress-stat-tile">
      <div className="progress-stat-value">{display}</div>
      <div className="progress-stat-label">{label}</div>
      {n === 0 && <div className="progress-stat-empty">no data yet</div>}
    </div>
  )
}

function BooksTable({ books }) {
  const { sorted, sort, dir, toggleSort } = useSortableRows(books, 'total')
  if (books.length === 0) {
    return <p className="progress-empty-state">Take some quizzes to see which books your vocabulary is strongest in.</p>
  }
  return (
    <table className="progress-table">
      <thead>
        <tr>
          <SortHeader label="Book" sortKey="title" sort={sort} dir={dir} onSort={toggleSort} />
          <SortHeader label="Author" sortKey="author" sort={sort} dir={dir} onSort={toggleSort} />
          <SortHeader label="Answers" sortKey="total" sort={sort} dir={dir} onSort={toggleSort} />
          <SortHeader label="Accuracy" sortKey="accuracy_pct" sort={sort} dir={dir} onSort={toggleSort} />
        </tr>
      </thead>
      <tbody>
        {sorted.map((b) => (
          <tr key={b.book_id}>
            <td>{b.title}</td>
            <td>{b.author ?? 'Unknown'}</td>
            <td>{b.total}</td>
            <td>{b.accuracy_pct === null ? '—' : `${b.accuracy_pct.toFixed(0)}%`}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function StrugglingTable({ words }) {
  const { sorted, sort, dir, toggleSort } = useSortableRows(words, 'miss_rate')
  if (words.length === 0) {
    return <p className="progress-empty-state">No struggling words yet -- keep quizzing to build up history.</p>
  }
  return (
    <table className="progress-table">
      <thead>
        <tr>
          <SortHeader label="Word" sortKey="lemma" sort={sort} dir={dir} onSort={toggleSort} />
          <SortHeader label="Miss rate" sortKey="miss_rate" sort={sort} dir={dir} onSort={toggleSort} />
          <SortHeader label="Streak" sortKey="streak" sort={sort} dir={dir} onSort={toggleSort} />
          <SortHeader label="Last seen" sortKey="last_seen_at" sort={sort} dir={dir} onSort={toggleSort} />
        </tr>
      </thead>
      <tbody>
        {sorted.map((w) => (
          <tr key={w.word_id}>
            <td><Link to={`/app/words/${w.word_id}`}>{w.lemma}</Link></td>
            <td>{(w.miss_rate * 100).toFixed(0)}%</td>
            <td>{w.streak}</td>
            <td>{w.last_seen_at ? new Date(w.last_seen_at).toLocaleDateString() : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Progress() {
  const [overview, setOverview] = useState(null)
  const [books, setBooks] = useState(null)
  const [struggling, setStruggling] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/progress/overview`).then((r) => (r.ok ? r.json() : Promise.reject(new Error('failed to load overview')))),
      fetch(`${API_BASE}/api/progress/books`).then((r) => (r.ok ? r.json() : Promise.reject(new Error('failed to load books')))),
      fetch(`${API_BASE}/api/progress/struggling`).then((r) => (r.ok ? r.json() : Promise.reject(new Error('failed to load struggling words')))),
    ])
      .then(([ov, bk, st]) => {
        setOverview(ov)
        setBooks(bk)
        setStruggling(st)
      })
      .catch((err) => setError(err.message))
  }, [])

  if (error) return <div className="error-banner">{error}</div>
  if (!overview || !books || !struggling) return <div className="page-loading">Loading…</div>

  return (
    <div className="progress-page">
      <header className="progress-header">
        <h1>Your progress</h1>
      </header>

      <section className="progress-tiles">
        {overview.tiles.map((t) => (
          <StatTile key={t.label} {...t} />
        ))}
      </section>

      <section className="progress-section">
        <div className="progress-section-head">
          <h2>Score over time</h2>
          <Link to="/app/quiz/history" className="progress-history-link">View full quiz history →</Link>
        </div>
        <SparkTrendLine points={overview.trend} />
      </section>

      <section className="progress-section progress-two-col">
        <div>
          <h2>Accuracy by question type</h2>
          <SingleHueBarChart buckets={overview.by_question_type} />
        </div>
        <div>
          <h2>Accuracy by domain</h2>
          <SingleHueBarChart buckets={overview.by_domain} />
          <p className="progress-hint">A word can belong to more than one domain, so totals may overlap.</p>
        </div>
      </section>

      <section className="progress-section">
        <h2>Accuracy by source book</h2>
        <BooksTable books={books} />
        <p className="progress-hint">A word can appear in more than one book, so answer counts may overlap across rows.</p>
      </section>

      <section className="progress-section">
        <h2>Struggling words</h2>
        <StrugglingTable words={struggling} />
      </section>
    </div>
  )
}

export default Progress
