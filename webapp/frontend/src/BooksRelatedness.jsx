import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import ClusterDendrogram from './ClusterDendrogram'
import ClusterMap from './ClusterMap'
import BookMatrix from './BookMatrix'
import ClusterHighlightSelect from './ClusterHighlightSelect'
import CorpusMap from './CorpusMap'
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
  { id: 'everything', label: 'Everything' },
]

// Module-level so CorpusMap's fetch effect sees stable functions.
const bookRelatedUrl = (id) => `/api/browse/books/${id}/related?top_k=20&k2=0`
const bookSharedUrl = (a, b) => `/api/browse/books/${a}/shared-words/${b}`

const SCOPES = [
  { id: 'volume', label: 'Most-represented' },
  { id: 'fame', label: 'Most famous' },
]

// Book-level counterpart to AuthorsRelatedness.jsx -- the top-N books at
// once (N = compute_book_clustering's own --top-n, 200 by default -- NOT
// literally every book; the header count below is fetched live, not a
// hardcoded guess). Same four tabs, all reading one shared clustering run
// (see compute_book_clustering): Map (default), Matrix, Dendrogram, and
// Graph (the force-directed view, reading book_similarity directly rather
// than the clustering run -- see books_relatedness's own docstring).
//
// scope picks WHICH node set every tab draws from: "volume" is the
// original top-N-by-word-count selection, "fame" is every
// book_fame.fame_score >= 8 book instead. Map/Matrix/Dendrogram read it
// from the correspondingly named clustering run (book_cluster vs.
// book_cluster_fame -- see compute_book_clustering's min_fame docstring);
// Graph reads it from books_relatedness's own scope param, which reuses
// the SAME book_cluster_fame node list but still pulls edges from
// book_similarity directly (that endpoint was never derived from the
// clustering run's precomputed grid, unlike the other three).
function BooksRelatedness() {
  const navigate = useNavigate()
  // The tab lives in the URL (?tab=everything) so a link can open straight
  // onto it and a refresh stays put.
  const [searchParams, setSearchParams] = useSearchParams()
  const tab = TABS.some((t) => t.id === searchParams.get('tab')) ? searchParams.get('tab') : 'map'
  const setTab = (id) => setSearchParams(id === 'map' ? {} : { tab: id }, { replace: true })
  const [scope, setScope] = useState('volume')
  const [clusterNodes, setClusterNodes] = useState(null) // null = loading
  // Owned here, not per-tab -- same reasoning as AuthorsRelatedness's own
  // `highlighted`: picking a book to highlight should survive switching
  // tabs. Holds just the book id -- ClusterHighlightSelect looks up
  // display fields from `items` itself.
  const [highlighted, setHighlighted] = useState(null)

  useEffect(() => {
    setClusterNodes(null)
    fetch(`${API_BASE}/api/browse/books/map?scope=${scope}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setClusterNodes(data ? data.nodes : []))
      .catch(() => setClusterNodes([]))
  }, [scope])

  // ClusterHighlightSelect's options -- restricted to books actually IN
  // this clustering run (the same set the Map/Matrix/Dendrogram tabs draw
  // from), not the whole ~11,000-book corpus -- same reasoning as
  // AuthorsRelatedness's own highlightItems.
  const highlightItems = useMemo(
    () =>
      (clusterNodes ?? [])
        .map((n) => ({ id: n.id, label: n.title, sublabel: n.author }))
        .sort((a, b) => a.label.localeCompare(b.label)),
    [clusterNodes],
  )

  function goToBook(book) {
    navigate(`/app/authors/${encodeURIComponent(book.author || '')}/${book.id}/relatedness`)
  }

  return (
    <div className="authors-page">
      <header className="authors-header">
        <div>
          <h1>{tab === 'everything' ? 'Every book' : clusterNodes ? `Top ${clusterNodes.length} books` : 'Top books'}</h1>
          <p className="muted">
            {tab === 'everything'
              ? 'The whole corpus, grouped by shared vocabulary'
              : scope === 'fame'
              ? 'Vocabulary overlap across the most historically important books (fame score ≥ 8)'
              : 'Vocabulary overlap across the most-represented books (by word count)'}
          </p>
        </div>
        <Link to="/app/books" className="authors-back-link">
          ← All books
        </Link>
      </header>

      <div className="authors-relatedness-controls">
        {tab !== 'everything' && <div className="graph-signal-toggle" role="group" aria-label="Selection">
          {SCOPES.map((s) => (
            <button key={s.id} type="button" className={scope === s.id ? 'active' : ''} onClick={() => setScope(s.id)}>
              {s.label}
            </button>
          ))}
        </div>}
        <div className="graph-signal-toggle" role="group" aria-label="View">
          {TABS.map((t) => (
            <button key={t.id} type="button" className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
        {tab !== 'everything' && <div className="authors-relatedness-highlight">
          <span className="muted">Highlight:</span>
          <ClusterHighlightSelect items={highlightItems} value={highlighted} onChange={setHighlighted} />
        </div>}
      </div>

      {tab === 'map' && <ClusterMap entity="book" highlightKey={highlighted} onItemClick={goToBook} scope={scope} />}

      {tab === 'matrix' && <BookMatrix highlightBookId={highlighted} scope={scope} />}

      {tab === 'dendrogram' && (
        <ClusterDendrogram entity="book" highlightKey={highlighted} onItemClick={goToBook} scope={scope} />
      )}

      {tab === 'everything' && (
        <CorpusMap
          entity="book"
          relatedUrl={bookRelatedUrl}
          sharedWordsUrl={bookSharedUrl}
          onOpen={(pts, i) => goToBook({ id: pts.key[i], author: pts.author[i] })}
        />
      )}

      {tab === 'graph' && (
        <RelatednessGraph
          initialId="__all__"
          highlightId={highlighted}
          scope={scope}
          fetchUrl={(_id, topK) => `${API_BASE}/api/browse/books/relatedness?top_k=${topK}&scope=${scope}`}
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
