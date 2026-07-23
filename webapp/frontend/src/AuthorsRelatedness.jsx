import { Link, useNavigate } from 'react-router-dom'
import RelatednessGraph from './RelatednessGraph'
import './Authors.css'

const API_BASE = ''

// Secondary page (per the relatedness plan -- lower priority than the
// per-author drilldown): every author at once, tractable because there are
// only dozens of them. RelatednessGraph is fetched once here with no
// per-author center -- fetchUrl ignores its `id` argument and always hits
// the global endpoint.
function AuthorsRelatedness() {
  const navigate = useNavigate()

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
    </div>
  )
}

export default AuthorsRelatedness
