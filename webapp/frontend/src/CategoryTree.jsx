import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { colorForBucket } from './domainColors'
import './CategoryTree.css'

const API_BASE = ''
const ROW_H = 30
const TREE_W = 180
const TREE_PAD = 6

// Every leaf USAS category, clustered by meaning (see categories_dendrogram
// in browse.py), drawn as a left-to-right dendrogram whose leaves line up
// one-for-one with HTML rows on the right: code, name, word count, then the
// category's words in book-count order, as many as fit on the row.
//
// Hand-rolled SVG like BookDendrogram.jsx -- elbow connectors, merge height
// on x (root at the left edge, leaves flush right against their rows).
function layoutTree(tree, rowIndexByCode) {
  const maxDist = tree.distance || 1
  const nodes = []
  function place(node) {
    if (node.code) {
      const i = rowIndexByCode[node.code]
      const n = { x: TREE_W, y: i * ROW_H + ROW_H / 2, lo: i, hi: i }
      nodes.push({ ...n, leaf: true })
      return n
    }
    const a = place(node.left)
    const b = place(node.right)
    const x = TREE_PAD + (TREE_W - TREE_PAD) * (1 - node.distance / maxDist)
    const n = { x, y: (a.y + b.y) / 2, lo: Math.min(a.lo, b.lo), hi: Math.max(a.hi, b.hi) }
    nodes.push({ ...n, a, b })
    return n
  }
  place(tree)
  return nodes
}

// The tree + rows alone -- embedded under the Categories page's overlap
// graph, and wrapped by the standalone page below.
export function CategoryTreeView() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [hovered, setHovered] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/categories/dendrogram`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`request failed (${r.status})`))))
      .then(setData)
      .catch((err) => setError(err.message || 'failed to load categories'))
  }, [])

  const nodes = useMemo(() => {
    if (!data?.tree) return []
    const index = Object.fromEntries(data.leaves.map((l, i) => [l.code, i]))
    return layoutTree(data.tree, index)
  }, [data])

  if (error) return <div className="error-banner">{error}</div>
  if (!data) return <div className="page-loading">Clustering categories…</div>
  if (!data.leaves.length) return <div className="page-loading">No categorized words yet.</div>

  const height = data.leaves.length * ROW_H
  // A merge is "on the hovered path" when the hovered leaf falls inside it.
  const onPath = (n) => hovered !== null && n.lo <= hovered && hovered <= n.hi

  return (
    <div className="cattree-body" style={{ '--row-h': `${ROW_H}px` }}>
      <svg
        className="cattree-svg"
        width={TREE_W + 2}
        height={height}
        viewBox={`0 0 ${TREE_W + 2} ${height}`}
        aria-hidden="true"
      >
        {nodes.filter((n) => !n.leaf).map((n, i) => {
          const hot = onPath(n)
          return (
            <g key={i} className={hot ? 'cattree-link is-hot' : 'cattree-link'}>
              <path d={`M${n.x},${n.a.y} V${n.b.y}`} />
              <path d={`M${n.x},${n.a.y} H${n.a.x}`} />
              <path d={`M${n.x},${n.b.y} H${n.b.x}`} />
            </g>
          )
        })}
      </svg>

      <ol className="cattree-rows">
        {data.leaves.map((leaf, i) => {
          const color = colorForBucket(leaf.bucket)
          const more = leaf.word_count - leaf.words.length
          const listUrl = `/app/words?${new URLSearchParams({
            top_code: leaf.code,
            sort: 'book_count',
            dir: 'desc',
          })}`
          return (
            <li
              key={leaf.code}
              className={hovered === i ? 'cattree-row is-hot' : 'cattree-row'}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
            >
              <span className="cattree-swatch" style={{ background: color }} />
              <span className="cattree-code" style={{ color }}>{leaf.code}</span>
              <span className="cattree-name" title={leaf.name}>{leaf.name}</span>
              <span className="cattree-count">{leaf.word_count.toLocaleString()}</span>
              <span className="cattree-words">
                {leaf.words.map((w, j) => (
                  <span key={w.id}>
                    {j > 0 && <span className="cattree-sep" aria-hidden="true"> · </span>}
                    <Link
                      to={`/app/words/${w.id}`}
                      className="cattree-word"
                      title={`${w.lemma} — in ${w.book_count.toLocaleString()} book${w.book_count === 1 ? '' : 's'}`}
                    >
                      {w.lemma}
                    </Link>
                  </span>
                ))}
              </span>
              <Link to={listUrl} className="cattree-more" title={`All ${leaf.word_count} words, most-used first`}>
                {more > 0 ? `all ${leaf.word_count.toLocaleString()}` : 'list'} <span aria-hidden="true">→</span>
              </Link>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

export const CATEGORY_TREE_BLURB =
  'Every finest-grained subject category, clustered by what its words mean, so related fields ' +
  'sit together even when the USAS scheme files them apart. Each row lists the category\u2019s ' +
  'words, those used in the most books first.'

function CategoryTree() {
  return (
    <div className="cattree-page">
      <header className="cattree-header">
        <Link to="/app/visualizations" className="cattree-crumb">← Visualizations</Link>
        <h1 className="cattree-title">Categories by meaning</h1>
        <p className="cattree-lede">{CATEGORY_TREE_BLURB}</p>
      </header>
      <CategoryTreeView />
    </div>
  )
}

export default CategoryTree
