import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import AuthorClusterMap from './AuthorClusterMap'
import AuthorDendrogram from './AuthorDendrogram'
import AuthorMatrix from './AuthorMatrix'
import RelatednessGraph from './RelatednessGraph'
import './Authors.css'
import './GraphView.css' // .graph-signal-toggle, reused here directly for the tab strip

const API_BASE = ''

const TABS = [
  { id: 'map', label: 'Map' },
  { id: 'matrix', label: 'Matrix' },
  { id: 'dendrogram', label: 'Dendrogram' },
  { id: 'graph', label: 'Graph' },
]

// Secondary page (per the relatedness plan -- lower priority than the
// per-author drilldown): every author at once, tractable because there are
// only dozens of them (well, the top few hundred by book count -- see
// compute_author_clustering's own top_n). Four tabs, all reading the same
// underlying clustering run (see compute_author_clustering): Map (default
// -- position/color are principled, derived from real MDS/clustering over
// the similarity structure, unlike the force graph's physics-simulation
// compromise layout), Matrix (precise pairwise lookup, seriated so related
// authors form visible blocks), Dendrogram (the clearest hierarchical
// narrative), and Graph (the original force-directed view, kept -- still
// a valid lower-priority option, not deleted).
function AuthorsRelatedness() {
  const navigate = useNavigate()
  const [tab, setTab] = useState('map')

  return (
    <div className="authors-page">
      <header className="authors-header">
        <div>
          <h1>All authors</h1>
          <p className="muted">Vocabulary overlap across every author</p>
        </div>
        <Link to="/app/authors" className="authors-back-link">
          ← All authors
        </Link>
      </header>

      <div className="graph-signal-toggle" role="group" aria-label="View">
        {TABS.map((t) => (
          <button key={t.id} type="button" className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'map' && (
        <AuthorClusterMap onAuthorClick={(author) => navigate(`/app/authors/${encodeURIComponent(author)}/relatedness`)} />
      )}

      {tab === 'matrix' && <AuthorMatrix />}

      {tab === 'dendrogram' && (
        <AuthorDendrogram onAuthorClick={(author) => navigate(`/app/authors/${encodeURIComponent(author)}/relatedness`)} />
      )}

      {tab === 'graph' && (
        <RelatednessGraph
          initialId="__all__"
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
