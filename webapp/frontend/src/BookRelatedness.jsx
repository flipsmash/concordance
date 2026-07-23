import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import RelatednessGraph from './RelatednessGraph'
import './Authors.css'
import './WorkDetail.css'

const API_BASE = ''

// Level 4 of the author drilldown: one book's full vocabulary-overlap
// relatedness graph. Mirrors WorkDetail.jsx's header/back-link shell; the
// graph itself is RelatednessGraph, copy-adapted from GraphView.jsx for
// lexical overlap (shared active words) instead of semantic embeddings --
// see the book/author relatedness plan.
function BookRelatedness() {
  const { author, bookId } = useParams()
  const navigate = useNavigate()
  const [book, setBook] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/books?book_id=${bookId}`)
      .then((res) => res.json())
      .then((data) => setBook(data.items[0] || null))
      .catch(() => {})
  }, [bookId])

  return (
    <div className="browse-page work-detail-page">
      <header className="authors-header">
        <div>
          <h1>{book ? book.title : 'Loading…'}</h1>
          <p className="muted">Books with the most overlapping vocabulary</p>
        </div>
        <Link to={`/app/authors/${encodeURIComponent(author)}/${bookId}`} className="authors-back-link">
          ← Back to {book ? book.title : 'book'}
        </Link>
      </header>

      <RelatednessGraph
        initialId={Number(bookId)}
        fetchUrl={(id, topK) => `${API_BASE}/api/browse/books/${id}/related?top_k=${topK}`}
        getLabel={(n) => n.title}
        getSublabel={(n) => n.author}
        onNodeNavigate={(node) =>
          navigate(`/app/authors/${encodeURIComponent(node.author || '')}/${node.id}/relatedness`)
        }
      />
    </div>
  )
}

export default BookRelatedness
