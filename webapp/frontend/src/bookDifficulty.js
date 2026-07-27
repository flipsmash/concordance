// Shared by AuthorWorks and Books -- both render the same BookRow shape
// from /api/browse/books and summarize its difficulty the same way. Sparse
// difficulty coverage is a known, real state of the corpus, shown honestly
// rather than papered over.
export function difficultySummary(book) {
  if (book.scored_word_count === 0) {
    return { stat: 'Not yet scored', qualifier: null }
  }
  const mean = book.mean_difficulty.toFixed(1)
  const stat =
    book.stddev_difficulty === null
      ? `${mean} difficulty (± N/A — not enough scored words)`
      : `${mean} ± ${book.stddev_difficulty.toFixed(1)} difficulty`
  return { stat, qualifier: `based on ${book.scored_word_count} of ${book.word_count} words` }
}

// unique extracted vocabulary words / this book's own distinct non-stopword
// count (concordance/archive_metadata.py's word_stats) -- how much of the
// book's real vocabulary is "assessed" advanced words, not filler.
export function densityLabel(density) {
  if (density == null) return null
  return `${(density * 100).toFixed(1)}% vocabulary density`
}

// A percent_rank blend of mean_difficulty and density (see browse.py's
// BookRow.overall_difficulty for the full formula) -- reads directly as a
// corpus-relative percentile, not a raw score on its own scale.
export function overallDifficultyLabel(overall) {
  if (overall == null) return null
  return `Harder than ${overall}% of the corpus`
}
