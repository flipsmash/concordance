import { useEffect, useMemo, useRef, useState } from 'react'
import SharedWordsPanel from './SharedWordsPanel'
import { cssVar } from './graphUtils'
import './AuthorMatrix.css'

const API_BASE = ''

// The seriated similarity matrix -- the same top-N authors as the cluster
// map, in hierarchical-clustering leaf order (see compute_author_clustering
// in concordance/db.py), so related authors form visible blocks instead of
// scattering across the grid. Canvas, not SVG: at N=200 that's 40,000
// cells, far past what individual SVG elements + React reconciliation
// should carry (the cluster map's 200 SVG circles are fine; 40,000 would
// not be). Row/column text labels are deliberately omitted -- 200 of them
// crammed along one axis would be illegible clutter -- hover reveals the
// pair's identity and exact numbers instead, and a click opens the shared-
// words comparison, since a matrix cell is inherently a two-entity pick in
// a way map/dendrogram node clicks (simple navigation) aren't.
function AuthorMatrix() {
  const [data, setData] = useState(null) // {authors, grid} | null (loading)
  const [error, setError] = useState('')
  const [hoverCell, setHoverCell] = useState(null) // {row, col} | null
  const [comparePair, setComparePair] = useState(null)
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const [size, setSize] = useState(0)

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/authors/matrix`)
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

  const n = data?.authors.length ?? 0
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
    const accent = cssVar('--accent', '#aa3bff')
    ctx.fillStyle = accent
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const cell = data.grid[i][j]
        ctx.globalAlpha = i === j ? 0.08 : Math.max(0, Math.min(1, cell.score))
        ctx.fillRect(j * cellPx, i * cellPx, Math.ceil(cellPx), Math.ceil(cellPx))
      }
    }
    ctx.globalAlpha = 1
  }, [data, size, n, cellPx])

  const hoverInfo = useMemo(() => {
    if (!hoverCell || !data) return null
    const { row, col } = hoverCell
    return { a: data.authors[row], b: data.authors[col], cell: data.grid[row][col] }
  }, [hoverCell, data])

  function handleMouseMove(e) {
    if (!data || cellPx === 0) return
    const rect = canvasRef.current.getBoundingClientRect()
    const col = Math.floor((e.clientX - rect.left) / cellPx)
    const row = Math.floor((e.clientY - rect.top) / cellPx)
    if (row >= 0 && row < n && col >= 0 && col < n) setHoverCell({ row, col })
  }

  function handleClick() {
    if (!hoverInfo || hoverInfo.a === hoverInfo.b) return
    setComparePair({ a: hoverInfo.a, b: hoverInfo.b })
  }

  const ready = data !== null && data.authors.length > 0

  return (
    <div className="author-matrix">
      {error && <div className="graph-error">{error}</div>}

      {/* Always mounted, even while loading/empty -- containerRef must be
          attached from first render for the ResizeObserver effect below
          (mount-only, empty deps) to ever find a real element. Making this
          wrapper conditional on `data` meant the observer's very first (and
          only) attempt to attach ran before data arrived, containerRef.current
          was still null, and the canvas stayed permanently 0x0 -- found live
          via Playwright: the canvas existed in the DOM but never became
          visible. */}
      <div className="author-matrix-canvas-wrap" ref={containerRef}>
        {data === null && !error && <div className="graph-loading">Loading…</div>}
        {data !== null && data.authors.length === 0 && (
          <div className="graph-empty">No clustering data yet — run `concordance author-clustering`.</div>
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
        <p className="author-matrix-tooltip muted">
          {hoverInfo && hoverInfo.a !== hoverInfo.b ? (
            <>
              <strong>{hoverInfo.a}</strong> × <strong>{hoverInfo.b}</strong> —{' '}
              {hoverInfo.cell.shared_word_count} shared words, {(hoverInfo.cell.score * 100).toFixed(0)}% overlap
              (click to compare)
            </>
          ) : (
            'Hover a cell to see the pair; click to compare their shared vocabulary.'
          )}
        </p>
      )}

      {comparePair && (
        <SharedWordsPanel
          fetchUrl={`/api/browse/authors/${encodeURIComponent(comparePair.a)}/shared-words/${encodeURIComponent(comparePair.b)}`}
          titleA={comparePair.a}
          titleB={comparePair.b}
          onClose={() => setComparePair(null)}
        />
      )}
    </div>
  )
}

export default AuthorMatrix
