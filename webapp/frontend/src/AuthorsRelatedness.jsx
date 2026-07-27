import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import AuthorClusterMap from './AuthorClusterMap'
import AuthorDendrogram from './AuthorDendrogram'
import AuthorMatrix from './AuthorMatrix'
import ClusterHighlightSelect from './ClusterHighlightSelect'
import RelatednessGraph from './RelatednessGraph'
import './Authors.css'
import './Browse.css' // .author-select* -- reused here for the highlight-an-author control
import './GraphView.css' // .graph-signal-toggle, reused here directly for the tab strip

const API_BASE = ''

const TABS = [
  { id: 'map', label: 'Map' },
  { id: 'matrix', label: 'Matrix' },
  { id: 'dendrogram', label: 'Dendrogram' },
  { id: 'graph', label: 'Graph' },
]

// Secondary page (per the relatedness plan -- lower priority than the
// per-author drilldown): the top-N authors at once (N = compute_author_
// clustering's own --top-n, 200 by default -- NOT literally every author;
// the header count below is fetched live, not a hardcoded guess, so it
// can't drift if that default is ever changed). Four tabs, all reading the
// same underlying clustering run (see compute_author_clustering): Map
// (default -- position/color are principled, derived from real MDS/
// clustering over the similarity structure, unlike the force graph's
// physics-simulation compromise layout), Matrix (precise pairwise lookup,
// seriated so related authors form visible blocks), Dendrogram (the
// clearest hierarchical narrative), and Graph (the original force-directed
// view, kept -- still a valid lower-priority option, not deleted).
function AuthorsRelatedness() {
  const navigate = useNavigate()
  const [tab, setTab] = useState('map')
  const [clusterNodes, setClusterNodes] = useState(null) // null = loading
  // Owned here, not per-tab -- picking an author to highlight should survive
  // switching tabs, since "where does X sit in this structure" is the same
  // question across map/matrix/dendrogram/graph. Holds just the author name
  // (the clustering run's own id), not a whole object -- ClusterHighlightSelect
  // looks up display fields from `items` itself.
  const [highlighted, setHighlighted] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/authors/map`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setClusterNodes(data ? data.nodes : []))
      .catch(() => setClusterNodes([]))
  }, [])

  // ClusterHighlightSelect's options -- restricted to authors actually IN
  // this clustering run (the same set the Map/Matrix/Dendrogram tabs draw
  // from), not the whole ~3,500-author corpus: searching the full corpus
  // here meant the overwhelming majority of picks landed on "isn't in this
  // view's top authors," a highlight control that mostly didn't work.
  const highlightItems = useMemo(
    () => (clusterNodes ?? []).map((n) => ({ id: n.author, label: n.author })).sort((a, b) => a.label.localeCompare(b.label)),
    [clusterNodes],
  )

  return (
    <div className="authors-page">
      <header className="authors-header">
        <div>
          <h1>{clusterNodes ? `Top ${clusterNodes.length} authors` : 'Top authors'}</h1>
          <p className="muted">
            Vocabulary overlap across the most-represented authors (by book count)
          </p>
        </div>
        <Link to="/app/authors" className="authors-back-link">
          ← All authors
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
          <ClusterHighlightSelect items={highlightItems} value={highlighted} onChange={setHighlighted} />
        </div>
      </div>

      {tab === 'map' && (
        <AuthorClusterMap
          highlightAuthor={highlighted}
          onAuthorClick={(author) => navigate(`/app/authors/${encodeURIComponent(author)}/relatedness`)}
        />
      )}

      {tab === 'matrix' && <AuthorMatrix highlightAuthor={highlighted} />}

      {tab === 'dendrogram' && (
        <AuthorDendrogram
          highlightAuthor={highlighted}
          onAuthorClick={(author) => navigate(`/app/authors/${encodeURIComponent(author)}/relatedness`)}
        />
      )}

      {tab === 'graph' && (
        <RelatednessGraph
          initialId="__all__"
          highlightId={highlighted}
          fetchUrl={(_id, topK) => `${API_BASE}/api/browse/authors/relatedness?top_k=${topK}`}
          getLabel={(n) => n.id}
          getSublabel={(n) => (n.book_count != null ? `${n.book_count} book${n.book_count === 1 ? '' : 's'}` : undefined)}
          onNodeNavigate={(node) => navigate(`/app/authors/${encodeURIComponent(node.id)}/relatedness`)}
          sharedWordsUrl={(a, b) =>
            `${API_BASE}/api/browse/authors/${encodeURIComponent(a)}/shared-words/${encodeURIComponent(b)}`
          }
        />
      )}
    </div>
  )
}

export default AuthorsRelatedness
