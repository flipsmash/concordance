import { Link, useNavigate, useParams } from 'react-router-dom'
import RelatednessGraph from './RelatednessGraph'
import './Authors.css'

const API_BASE = ''

// Level 3 of the author drilldown (a sibling of AuthorWorks, not nested
// under a specific book): one author's full vocabulary-overlap relatedness
// graph. See the book/author relatedness plan -- RelatednessGraph is the
// same component BookRelatedness.jsx uses, just pointed at the author
// endpoints and string ids instead of book ids.
function AuthorRelatedness() {
  const { author } = useParams()
  const navigate = useNavigate()

  return (
    <div className="authors-page">
      <header className="authors-header">
        <div>
          <h1>{author}</h1>
          <p className="muted">Authors with the most overlapping vocabulary</p>
        </div>
        <Link to={`/app/authors/${encodeURIComponent(author)}`} className="authors-back-link">
          ← {author}'s works
        </Link>
      </header>

      <RelatednessGraph
        initialId={author}
        fetchUrl={(id, topK) => `${API_BASE}/api/browse/authors/${encodeURIComponent(id)}/related?top_k=${topK}`}
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

export default AuthorRelatedness
