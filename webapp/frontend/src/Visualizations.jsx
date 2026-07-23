import { Suspense, lazy } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import AuthorSelect from './AuthorSelect'
import BookSelect from './BookSelect'
import './Browse.css'
import './Visualizations.css'

// Lazy for the same reason App.jsx already lazy-loads GraphView: pulls in
// react-force-graph-2d's canvas/d3-force bundle only once this page is
// actually opened.
const GraphView = lazy(() => import('./GraphView'))

// A hub for every relatedness/graph view the app has, all of which
// otherwise only exist as links buried behind a specific word/book/author's
// own page (or, for the global authors graph, no link at all until this
// page). Three ways in, one per granularity: words by meaning (embedded,
// full search), books by vocabulary overlap (pick one, land on its ego
// graph), authors by vocabulary overlap (pick one OR see all of them at
// once).
function Visualizations() {
  const navigate = useNavigate()

  return (
    <div className="browse-page">
      <header className="browse-header">
        <h1>Visualizations</h1>
        <Link to="/app" className="browse-quiz-link">← Back to browse</Link>
      </header>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Books by vocabulary overlap</h2>
        <p className="viz-description">
          Pick a book to see the other books that share the most rare vocabulary with it.
        </p>
        <BookSelect
          placeholder="Search for a book…"
          onPick={(book) => navigate(`/app/authors/${encodeURIComponent(book.author || '')}/${book.id}/relatedness`)}
        />
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Authors by vocabulary overlap</h2>
        <p className="viz-description">
          Pick an author to see who shares the most rare vocabulary with them, or view every
          author at once.
        </p>
        <div className="viz-author-row">
          <AuthorSelect
            value={null}
            onChange={(author) => author && navigate(`/app/authors/${encodeURIComponent(author)}/relatedness`)}
          />
          <Link to="/app/authors/relatedness" className="browse-quiz-link">
            See all authors at once →
          </Link>
        </div>
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Words by meaning</h2>
        <p className="viz-description">
          Search for a word to see its nearest neighbors by definition or spelling.
        </p>
        <div className="viz-graph-embed">
          <Suspense fallback={<div className="page-loading">Loading…</div>}>
            <GraphView onNodeNavigate={(node) => navigate(`/app/words/${node.id}`)} />
          </Suspense>
        </div>
      </section>
    </div>
  )
}

export default Visualizations
