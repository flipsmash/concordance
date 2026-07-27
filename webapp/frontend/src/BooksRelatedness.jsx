import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import BookClusterMap from './BookClusterMap'
import BookDendrogram from './BookDendrogram'
import BookMatrix from './BookMatrix'
import BookSelect from './BookSelect'
import RelatednessGraph from './RelatednessGraph'
import './Authors.css'
import './Browse.css' // .author-select* -- reused here for the highlight-a-book control
import './GraphView.css' // .graph-signal-toggle, reused here directly for the tab strip

const API_BASE = ''

const TABS = [
  { id: 'map', label: 'Map' },
  { id: 'matrix', label: 'Matrix' },
  { id: 'dendrogram', label: 'Dendrogram' },
  { id: 'graph', label: 'Graph' },
]

// Book-level counterpart to AuthorsRelatedness.jsx -- every book at once,
// tractable because compute_book_clustering restricts the map/matrix/
// dendrogram to the top_n (by word count) books, same as author-clustering
// does one level up. Same four tabs, all reading one shared clustering run
// (see compute_book_clustering): Map (default), Matrix, Dendrogram, and
// Graph (the force-directed view, reading book_similarity directly rather
// than the clustering run -- see books_relatedness's own docstring).
function BooksRelatedness() {
  const navigate = useNavigate()
  const [tab, setTab] = useState('map')
  // Owned here, not per-tab -- same reasoning as AuthorsRelatedness's own
  // `highlighted`: picking a book to highlight should survive switching
  // tabs. Holds the full book object (BookSelect's controlled value), not
  // just an id, so the chosen-chip can show its title.
  const [highlighted, setHighlighted] = useState(null)
  const highlightBookId = highlighted?.id ?? null

  function goToBook(book) {
    navigate(`/app/authors/${encodeURIComponent(book.author || '')}/${book.id}/relatedness`)
  }

  return (
    <div className="authors-page">
      <header className="authors-header">
        <div>
          <h1>All books</h1>
          <p className="muted">Vocabulary overlap across every book</p>
        </div>
        <Link to="/app/books" className="authors-back-link">
          ← All books
        </Link>
      </header>

      <div className="authors-relatedness-controls">
        <div className="graph-signal-toggle" role="group" aria-label="View">
          {TABS.map((t) => (
            <button key={t.id} type="button" className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
        <div className="authors-relatedness-highlight">
          <span className="muted">Highlight:</span>
          <BookSelect value={highlighted} onChange={setHighlighted} />
        </div>
      </div>

      {tab === 'map' && <BookClusterMap highlightBookId={highlightBookId} onBookClick={goToBook} />}

      {tab === 'matrix' && <BookMatrix highlightBookId={highlightBookId} />}

      {tab === 'dendrogram' && <BookDendrogram highlightBookId={highlightBookId} onBookClick={goToBook} />}

      {tab === 'graph' && (
        <RelatednessGraph
          initialId="__all__"
          highlightId={highlightBookId}
          fetchUrl={(_id, topK) => `${API_BASE}/api/browse/books/relatedness?top_k=${topK}`}
          getLabel={(n) => n.title}
          getSublabel={(n) => n.author ?? undefined}
          onNodeNavigate={goToBook}
          sharedWordsUrl={(a, b) => `${API_BASE}/api/browse/books/${a}/shared-words/${b}`}
        />
      )}
    </div>
  )
}

export default BooksRelatedness
