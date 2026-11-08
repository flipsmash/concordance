import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import DifficultyHistogram from './DifficultyHistogram'
import { VizPage, VizPageHeader } from './VizPage'
import './Visualizations.css'

const API_BASE = ''

// UniqueWordBucket ({label, count}) -> DifficultyHistogram's bar shape. The
// band fields only drive its selected/"unscored" styling, so placeholders do.
function toHistBands(buckets) {
  return buckets.map((b) => ({ band_min: 0, band_max: 0, label: b.label, word_count: b.count }))
}

// Same, keeping band_min === null on "Not yet scored" (its greyed styling);
// an empty "Not yet scored" bar is noise, so it goes.
function toFameHistBands(buckets) {
  return buckets.filter((b) => !(b.label === 'Not yet scored' && !b.count)).map((b) => ({
    band_min: b.label === 'Not yet scored' ? null : 0,
    band_max: 0,
    label: b.label,
    word_count: b.count,
  }))
}

// An empty "Not enough data" difficulty bar is noise too.
function dropEmptyUnscored(bands) {
  return bands.filter((b) => !(b.band_min === null && !b.word_count))
}

// A fame bar's label carries the real score -- enough for the deep link.
function fameHistQuery(label) {
  return label === 'Not yet scored' ? { fame_unscored_only: 'true' } : { fame_min: label, fame_max: label }
}

function useHist(url, adapt = (x) => x) {
  const [bands, setBands] = useState(null) // null = loading
  useEffect(() => {
    fetch(`${API_BASE}${url}`)
      .then((r) => r.json())
      .then((d) => setBands(adapt(d)))
      .catch(() => setBands([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url])
  return bands
}

// One measure, books and authors side by side.
function Pair({ title, children, book, author }) {
  return (
    <section className="viz-dist">
      <h2 className="viz-dist-title">{title}</h2>
      <p className="viz-dist-text">{children}</p>
      <div className="viz-dist-pair">
        <div>
          <h3 className="viz-dist-sub">Books</h3>
          {book}
        </div>
        <div>
          <h3 className="viz-dist-sub">Authors</h3>
          {author}
        </div>
      </div>
    </section>
  )
}

function VizDistributions() {
  const navigate = useNavigate()
  const bookUnique = useHist('/api/browse/unique-word-histogram?scope=book', toHistBands)
  const authorUnique = useHist('/api/browse/unique-word-histogram?scope=author', toHistBands)
  const bookFame = useHist('/api/browse/fame-histogram?scope=book', toFameHistBands)
  const authorFame = useHist('/api/browse/fame-histogram?scope=author', toFameHistBands)
  const bookDiff = useHist('/api/browse/overall-difficulty-histogram?scope=book', dropEmptyUnscored)
  const authorDiff = useHist('/api/browse/overall-difficulty-histogram?scope=author', dropEmptyUnscored)
  const go = (path) => navigate(path)

  return (
    <VizPage>
      <VizPageHeader title="How the books and authors spread out">
        How much vocabulary each book or author has to itself, how famous they are, and how hard their words are.
        Click any bar to list the books or authors in it.
      </VizPageHeader>

      <Pair
        title="Words found nowhere else"
        book={<DifficultyHistogram bands={bookUnique} selectedBand={null} unit="books"
          headline={(b, pct) => `${b.word_count.toLocaleString()} books (${pct}%) share every word with another book. The rest:`}
          onSelect={(b) => go(`/app/books?unique_word_bucket=${encodeURIComponent(b.label)}`)} />}
        author={<DifficultyHistogram bands={authorUnique} selectedBand={null} unit="authors"
          headline={(b, pct) => `${b.word_count.toLocaleString()} authors (${pct}%) share every word with another author. The rest:`}
          onSelect={(b) => go(`/app/authors?unique_word_bucket=${encodeURIComponent(b.label)}`)} />}
      >
        How many of their words appear in no other book (or no other author&rsquo;s books). Most share nearly
        everything; a few contribute dozens of words all their own.
      </Pair>

      <Pair
        title="Fame"
        book={<DifficultyHistogram bands={bookFame} selectedBand={null} unit="books"
          onSelect={(b) => go(`/app/books?${new URLSearchParams(fameHistQuery(b.label))}`)} />}
        author={<DifficultyHistogram bands={authorFame} selectedBand={null} unit="authors"
          onSelect={(b) => go(`/app/authors?${new URLSearchParams(fameHistQuery(b.label))}`)} />}
      >
        Historical and cultural importance on an absolute 1&ndash;10 scale (not a ranking): most scores sit low by
        design, so a bar at 8&ndash;10 means genuinely famous.
      </Pair>

      <Pair
        title="Difficulty"
        book={<DifficultyHistogram bands={bookDiff} selectedBand={null} unit="books"
          onSelect={(b) => go(`/app/books?${new URLSearchParams({ overall_difficulty_band: b.label })}`)} />}
        author={<DifficultyHistogram bands={authorDiff} selectedBand={null} unit="authors"
          onSelect={(b) => go(`/app/authors?${new URLSearchParams({ overall_difficulty_band: b.label })}`)} />}
      >
        A percentile within the collection, blending how hard a book&rsquo;s words are with how densely they
        occur &mdash; near 100 means harder than almost everything else here.
      </Pair>
    </VizPage>
  )
}

export default VizDistributions
