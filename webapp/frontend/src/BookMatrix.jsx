import { useEffect, useMemo, useRef, useState } from 'react'
import SharedWordsPanel from './SharedWordsPanel'
import { cssVar } from './graphUtils'
import './BookMatrix.css'
import './GraphView.css' // .graph-maximize -- reused here for the fullscreen button

const API_BASE = ''

// Book-level counterpart to AuthorMatrix.jsx -- see that file for the full
// rationale (canvas not SVG at N up to 200, hover-then-label, click opens
// shared-words). `data.books` is a list of {id, title, author} objects
// here, not plain strings like `data.authors` -- a title alone isn't a
// stable identity (two books can share one) or enough to build a
// shared-words URL, so every lookup below keys off book.id.
function BookMatrix({ highlightBookId }) {
  const [data, setData] = useState(null) // {books, grid} | null (loading)
  const [error, setError] = useState('')
  const [hoverCell, setHoverCell] = useState(null) // {row, col} | null
  const [comparePair, setComparePair] = useState(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const viewRef = useRef(null)
  const [size, setSize] = useState(0)

  useEffect(() => {
    function handleFullscreenChange() {
      setIsFullscreen(document.fullscreenElement === viewRef.current)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  function toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      viewRef.current?.requestFullscreen()
    }
  }

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/books/matrix`)
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then(setData)
      .catch((err) => setError(err.message || 'failed to load matrix'))
  }, [])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      setSize(Math.max(0, Math.min(width, height)))
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const n = data?.books.length ?? 0
  const cellPx = n > 0 ? size / n : 0

  useEffect(() => {
    if (!data || !canvasRef.current || size === 0 || n === 0) return
    const canvas = canvasRef.current
    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    const ctx = canvas.getContext('2d')
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, size, size)
    const accent = cssVar('--accent', '#1c6dbd')
    ctx.fillStyle = accent
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const cell = data.grid[i][j]
        ctx.globalAlpha = i === j ? 0.08 : Math.max(0, Math.min(1, cell.score))
        ctx.fillRect(j * cellPx, i * cellPx, Math.ceil(cellPx), Math.ceil(cellPx))
      }
    }
    ctx.globalAlpha = 1

    // Highlight-a-book-of-interest: outline its whole row + column band,
    // same reasoning as AuthorMatrix's own highlight (a single cell would
    // be nearly invisible at N up to 200; the full cross is unmistakable).
    const highlightIndex = highlightBookId != null ? data.books.findIndex((b) => b.id === highlightBookId) : -1
    if (highlightIndex >= 0) {
      ctx.strokeStyle = accent
      ctx.lineWidth = 2
      ctx.strokeRect(0, highlightIndex * cellPx, size, cellPx)
      ctx.strokeRect(highlightIndex * cellPx, 0, cellPx, size)
    }
  }, [data, size, n, cellPx, highlightBookId])

  const hoverInfo = useMemo(() => {
    if (!hoverCell || !data) return null
    const { row, col } = hoverCell
    return { a: data.books[row], b: data.books[col], cell: data.grid[row][col] }
  }, [hoverCell, data])

  function handleMouseMove(e) {
    if (!data || cellPx === 0) return
    const rect = canvasRef.current.getBoundingClientRect()
    const col = Math.floor((e.clientX - rect.left) / cellPx)
    const row = Math.floor((e.clientY - rect.top) / cellPx)
    if (row >= 0 && row < n && col >= 0 && col < n) setHoverCell({ row, col })
  }

  function handleClick() {
    if (!hoverInfo || hoverInfo.a.id === hoverInfo.b.id) return
    setComparePair({ a: hoverInfo.a, b: hoverInfo.b })
  }

  const ready = data !== null && data.books.length > 0
  const highlightBook = highlightBookId != null && data ? data.books.find((b) => b.id === highlightBookId) : null

  return (
    <div className="book-matrix" ref={viewRef}>
      <div className="graph-controls book-matrix-controls">
        <button type="button" className="graph-maximize" onClick={toggleFullscreen}>
          {isFullscreen ? 'Exit fullscreen' : 'Maximize'}
        </button>
      </div>
      {error && <div className="graph-error">{error}</div>}

      {/* Always mounted, even while loading/empty -- see AuthorMatrix's own
          comment: containerRef must be attached from first render for the
          ResizeObserver effect to ever find a real element. */}
      <div className="book-matrix-canvas-wrap" ref={containerRef}>
        {data === null && !error && <div className="graph-loading">Loading…</div>}
        {data !== null && data.books.length === 0 && (
          <div className="graph-empty">No clustering data yet — run `concordance book-clustering`.</div>
        )}
        {ready && (
          <canvas
            ref={canvasRef}
            style={{ width: size, height: size }}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHoverCell(null)}
            onClick={handleClick}
          />
        )}
      </div>
      {ready && (
        <p className="book-matrix-tooltip muted">
          {hoverInfo && hoverInfo.a.id !== hoverInfo.b.id ? (
            <>
              <strong>{hoverInfo.a.title}</strong> × <strong>{hoverInfo.b.title}</strong> —{' '}
              {hoverInfo.cell.shared_word_count} shared words, {(hoverInfo.cell.score * 100).toFixed(0)}% overlap
              (click to compare)
            </>
          ) : highlightBookId != null && highlightBook ? (
            <>
              Highlighting <strong>{highlightBook.title}</strong>'s row and column. Hover a cell to see a specific pair.
            </>
          ) : highlightBookId != null ? (
            `That book isn't in this matrix's top books.`
          ) : (
            'Hover a cell to see the pair; click to compare their shared vocabulary.'
          )}
        </p>
      )}

      {comparePair && (
        <SharedWordsPanel
          fetchUrl={`/api/browse/books/${comparePair.a.id}/shared-words/${comparePair.b.id}`}
          titleA={comparePair.a.title}
          titleB={comparePair.b.title}
          onClose={() => setComparePair(null)}
        />
      )}
    </div>
  )
}

export default BookMatrix
