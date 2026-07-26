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
