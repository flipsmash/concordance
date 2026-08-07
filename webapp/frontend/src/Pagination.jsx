import { useState } from 'react'
import './Pagination.css'

// 1 … (current-2..current+2) … X, collapsing the ellipsis away when there's
// no real gap to hide. delta=2 keeps a little context around the current
// page without the strip getting unwieldy.
//
// X is normally the last page, but a numbered button that just lands on
// the same page "Last »" already reaches is dead weight -- once the gap is
// big enough to matter, X becomes the page halfway through the remaining
// distance instead (still a real jump forward, and clicking it again keeps
// halving the distance to the end). A small gap (<=2 pages) just shows the
// real last page as before -- there's no meaningful "remaining distance"
// to offer a shortcut through.
function buildPageList(current, total) {
  const delta = 2
  const start = Math.max(2, current - delta)
  const end = Math.min(total - 1, current + delta)
  const pages = [1]
  if (start > 2) pages.push('…')
  for (let i = start; i <= end; i++) pages.push(i)
  if (end < total - 1) {
    pages.push('…')
    pages.push(total - end > 2 ? Math.round((end + total) / 2) : total)
  } else if (total > 1) {
    pages.push(total)
  }
  return pages
}

// Shared page-jump control for every large paginated list (words, books,
// authors) -- a numbered strip alone doesn't scale here: the word list runs
// into the thousands of pages, so clicking Next/Prev (or even the numbered
// strip's +/-2 window) to reach page 1500 from page 1 is impractical. The
// "Go to page" input is the actual answer to "jump around by pages at a
// time easily"; First/Last and the numbered strip cover the common nearby
// moves without a full page reload of typing.
function Pagination({ page, totalPages, total, itemLabel, onPageChange }) {
  const [jumpValue, setJumpValue] = useState('')

  function goTo(p) {
    const clamped = Math.max(1, Math.min(totalPages, p))
    if (clamped !== page) onPageChange(clamped)
  }

  function submitJump(e) {
    e.preventDefault()
    const n = parseInt(jumpValue, 10)
    if (!Number.isNaN(n)) goTo(n)
    setJumpValue('')
  }

  if (totalPages <= 1) {
    return (
      <footer className="pagination">
        <span className="pagination-summary">
          {total} {itemLabel}
        </span>
      </footer>
    )
  }

  const pages = buildPageList(page, totalPages)

  return (
    <footer className="pagination">
      <div className="pagination-controls">
        <button
          type="button"
          className="pagination-edge"
          disabled={page <= 1}
          onClick={() => goTo(1)}
          title="First page"
        >
          « First
        </button>
        <button type="button" disabled={page <= 1} onClick={() => goTo(page - 1)}>
          ← Prev
        </button>
        <div className="pagination-numbers">
          {pages.map((p, i) =>
            p === '…' ? (
              <span key={`ellipsis-${i}`} className="pagination-ellipsis">
                …
              </span>
            ) : (
              <button
                type="button"
                key={p}
                className={p === page ? 'pagination-page active' : 'pagination-page'}
                onClick={() => goTo(p)}
              >
                {p}
              </button>
            ),
          )}
        </div>
        <button type="button" disabled={page >= totalPages} onClick={() => goTo(page + 1)}>
          Next →
        </button>
        <button
          type="button"
          className="pagination-edge"
          disabled={page >= totalPages}
          onClick={() => goTo(totalPages)}
          title="Last page"
        >
          Last »
        </button>
      </div>

      <form className="pagination-jump" onSubmit={submitJump}>
        <span className="pagination-summary">
          Page {page} of {totalPages} ({total} {itemLabel})
        </span>
        <input
          type="number"
          min="1"
          max={totalPages}
          placeholder="Go to page…"
          value={jumpValue}
          onChange={(e) => setJumpValue(e.target.value)}
        />
        <button type="submit">Go</button>
      </form>
    </footer>
  )
}

export default Pagination
