import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import './Home.css'
import { LogoMark } from './Logo'

const API_BASE = ''

const TOC = [
  { label: 'Words', to: '/app/words' },
  { label: 'Books', to: '/app/books' },
  { label: 'Authors', to: '/app/authors' },
  { label: 'Categories', to: '/app/categories' },
  { label: 'Visualizations', to: '/app/visualizations' },
  { label: 'Quiz', to: '/app/quiz' },
  { label: 'Progress', to: '/app/progress' },
  { label: 'Sets', to: '/app/sets' },
]

const fmt = new Intl.NumberFormat('en-US')

function Num({ n }) {
  return <span className="home-num">{fmt.format(n)}</span>
}

function Home() {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/api/home/summary`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('failed to load summary'))))
      .then(setSummary)
      .catch((err) => setError(err.message))
  }, [])

  if (error) return <div className="error-banner">{error}</div>
  if (!summary) return <div className="page-loading">Loading…</div>

  const bonusParts = []
  if (summary.most_acclaimed_book) {
    bonusParts.push(
      <span key="fame">
        Most acclaimed: <em>{summary.most_acclaimed_book}</em>
        {summary.most_acclaimed_author ? ` by ${summary.most_acclaimed_author}` : ''}
      </span>
    )
  }
  if (summary.hardest_word) {
    bonusParts.push(<span key="hard">Hardest word: <em>{summary.hardest_word}</em></span>)
  }
  if (summary.quiz_questions_answered > 0) {
    bonusParts.push(
      <span key="quiz"><Num n={summary.quiz_questions_answered} /> quiz questions answered</span>
    )
  }

  return (
    <div className="home-page">
      <div className="home-title-block">
        <LogoMark width={340} className="home-logo" />
        <p className="home-subtitle">
          concordance, <em>n.</em> — a state of agreement or harmony, or an alphabetical list of the
          principal words in a body of work, with the passages in which they occur.
        </p>
      </div>

      <hr className="home-rule" />

      <p className="home-colophon">
        Drawn from <Num n={summary.total_books} /> books by <Num n={summary.total_authors} /> authors,
        comprising <Num n={summary.total_words} /> words across <Num n={summary.total_categories} /> categories.
      </p>

      {bonusParts.length > 0 && (
        <p className="home-bonus">
          {bonusParts.reduce((acc, part, i) => (i === 0 ? [part] : [...acc, ' · ', part]), [])}
        </p>
      )}

      <hr className="home-rule" />

      <div className="home-toc">
        <div className="home-toc-eyebrow">Contents</div>
        {TOC.map((t) => (
          <Link key={t.to} to={t.to} className="home-toc-row">
            <span className="home-toc-label">{t.label}</span>
            <span className="home-toc-leader" aria-hidden="true" />
            <span className="home-toc-arrow" aria-hidden="true">→</span>
          </Link>
        ))}
      </div>
    </div>
  )
}

export default Home
