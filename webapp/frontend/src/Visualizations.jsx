import { Suspense, lazy, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import AuthorSelect from './AuthorSelect'
import BookSelect from './BookSelect'
import DifficultyHistogram from './DifficultyHistogram'
import './Browse.css'
import './Visualizations.css'

const API_BASE = ''

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
// UniqueWordBucket ({label, count}) -> DifficultyHistogram's generic
// {band_min, band_max, label, word_count} bar shape. band_min/max only
// matter for DifficultyHistogram's click-to-filter highlighting and its
// "unscored" styling special-case (band_min === null) -- neither applies
// here (these bars aren't clickable filters), so any non-null placeholder
// works.
function toHistBands(buckets) {
  return buckets.map((b) => ({ band_min: 0, band_max: 0, label: b.label, word_count: b.count }))
}

// Same adapter as toHistBands, except the "Not yet scored" bucket keeps
// band_min === null -- that's what triggers DifficultyHistogram's distinct
// (greyed-out) styling for an unscored column, which toHistBands' buckets
// never have a use for since none of them carry an "unscored" concept.
function toFameHistBands(buckets) {
  return buckets.map((b) => ({
    band_min: b.label === 'Not yet scored' ? null : 0,
    band_max: 0,
    label: b.label,
    word_count: b.count,
  }))
}

// A fame bar's own label ("1".."10", or "Not yet scored") is enough to
// build the deep-link query on its own -- unlike band_min/band_max (both
// forced to 0/null above purely for DifficultyHistogram's styling hook),
// the label is the one field that still carries the real score.
function fameHistQuery(label) {
  return label === 'Not yet scored'
    ? { fame_unscored_only: 'true' }
    : { fame_min: label, fame_max: label }
}

function Visualizations() {
  const navigate = useNavigate()
  const [bookHist, setBookHist] = useState([])
  const [authorHist, setAuthorHist] = useState([])
  const [bookFameHist, setBookFameHist] = useState([])
  const [authorFameHist, setAuthorFameHist] = useState([])
  const [bookDifficultyHist, setBookDifficultyHist] = useState([])
  const [authorDifficultyHist, setAuthorDifficultyHist] = useState([])

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/unique-word-histogram?scope=book`)
      .then((res) => res.json())
      .then((data) => setBookHist(toHistBands(data)))
      .catch(() => {})
    fetch(`${API_BASE}/api/browse/unique-word-histogram?scope=author`)
      .then((res) => res.json())
      .then((data) => setAuthorHist(toHistBands(data)))
      .catch(() => {})
    fetch(`${API_BASE}/api/browse/fame-histogram?scope=book`)
      .then((res) => res.json())
      .then((data) => setBookFameHist(toFameHistBands(data)))
      .catch(() => {})
    fetch(`${API_BASE}/api/browse/fame-histogram?scope=author`)
      .then((res) => res.json())
      .then((data) => setAuthorFameHist(toFameHistBands(data)))
      .catch(() => {})
    // overall-difficulty-histogram already returns DifficultyHistogram's own
    // {band_min, band_max, label, word_count} shape (it buckets a real 0-100
    // percentile, same as word-level difficulty-bands) -- no toHistBands/
    // toFameHistBands adapter needed, unlike the two above.
    fetch(`${API_BASE}/api/browse/overall-difficulty-histogram?scope=book`)
      .then((res) => res.json())
      .then(setBookDifficultyHist)
      .catch(() => {})
    fetch(`${API_BASE}/api/browse/overall-difficulty-histogram?scope=author`)
      .then((res) => res.json())
      .then(setAuthorDifficultyHist)
      .catch(() => {})
  }, [])

  return (
    <div className="browse-page">
      <header className="browse-header">
        <h1>Visualizations</h1>
      </header>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Books by vocabulary overlap</h2>
        <p className="viz-description">
          Pick a book to see the other books that share the most rare vocabulary with it, or
          browse the top books at once.
        </p>
        <div className="viz-author-row">
          <BookSelect
            placeholder="Search for a book…"
            onPick={(book) => navigate(`/app/authors/${encodeURIComponent(book.author || '')}/${book.id}/relatedness`)}
          />
          <Link to="/app/books/relatedness" className="browse-quiz-link">
            See the top books at once →
          </Link>
        </div>
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Authors by vocabulary overlap</h2>
        <p className="viz-description">
          Pick an author to see who shares the most rare vocabulary with them, or browse the top
          authors at once.
        </p>
        <div className="viz-author-row">
          <AuthorSelect
            value={null}
            onChange={(author) => author && navigate(`/app/authors/${encodeURIComponent(author)}/relatedness`)}
          />
          <Link to="/app/authors/relatedness" className="browse-quiz-link">
            See the top authors at once →
          </Link>
        </div>
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Words unique to each book</h2>
        <p className="viz-description">
          How many books contribute vocabulary nothing else in the corpus does, and how many of
          their words that amounts to -- most books share nearly everything they use with at
          least one other book, but a few contribute dozens or more words all their own.
        </p>
        <DifficultyHistogram
          bands={bookHist}
          selectedBand={null}
          onSelect={(b) => navigate(`/app/books?unique_word_bucket=${encodeURIComponent(b.label)}`)}
        />
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Words unique to each author</h2>
        <p className="viz-description">
          Same question, aggregated across an author's whole body of work -- a word counts as
          theirs alone even if it appears in two of their own books, as long as no other author's
          book uses it.
        </p>
        <DifficultyHistogram
          bands={authorHist}
          selectedBand={null}
          onSelect={(b) => navigate(`/app/authors?unique_word_bucket=${encodeURIComponent(b.label)}`)}
        />
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Book fame distribution</h2>
        <p className="viz-description">
          Historical/cultural importance, an absolute 1–10 scale (not a percentile) where most
          scores sit low by design — only genuinely famous books score high, so a tall bar at 8–10
          means something. Fame is scored by a separate, manually-run process, not part of routine
          maintenance, so a large "Not yet scored" bar reflects that backlog, not missing data.
        </p>
        <DifficultyHistogram
          bands={bookFameHist}
          selectedBand={null}
          onSelect={(b) => navigate(`/app/books?${new URLSearchParams(fameHistQuery(b.label))}`)}
        />
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Author fame distribution</h2>
        <p className="viz-description">
          Same absolute 1–10 scale, aggregated per author rather than per book.
        </p>
        <DifficultyHistogram
          bands={authorFameHist}
          selectedBand={null}
          onSelect={(b) => navigate(`/app/authors?${new URLSearchParams(fameHistQuery(b.label))}`)}
        />
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Book difficulty distribution</h2>
        <p className="viz-description">
          Overall difficulty is a corpus-relative percentile (not an absolute score) blending a
          book's average word difficulty with how dense its rare vocabulary is -- a bar near 100
          means harder than nearly the whole corpus. "Not enough data" covers books missing either
          input (no scored words, or no distinct-word-count baseline to measure density against).
        </p>
        <DifficultyHistogram
          bands={bookDifficultyHist}
          selectedBand={null}
          onSelect={(b) =>
            navigate(`/app/books?${new URLSearchParams({ overall_difficulty_band: b.label })}`)
          }
        />
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Author difficulty distribution</h2>
        <p className="viz-description">
          Same percentile, aggregated per author rather than per book.
        </p>
        <DifficultyHistogram
          bands={authorDifficultyHist}
          selectedBand={null}
          onSelect={(b) =>
            navigate(`/app/authors?${new URLSearchParams({ overall_difficulty_band: b.label })}`)
          }
        />
      </section>

      <section className="browse-facets viz-section">
        <h2 className="viz-heading">Works &amp; authors by discipline category</h2>
        <p className="viz-description">
          See how vocabulary distributes across subject-matter fields (science, law, arts…) --
          works and authors with a similar mix sit close together, even with no words in common.
        </p>
        <Link to="/app/visualizations/domain-map" className="browse-quiz-link">
          Open the map →
        </Link>
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
