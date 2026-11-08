import { Suspense, lazy, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { VizPage, VizPageHeader } from './VizPage'
import './Visualizations.css'

const GraphView = lazy(() => import('./GraphView'))

// A random word to start from, so the page opens on a graph rather than an
// empty search box: the same random pick as "Surprise me", retried a few
// times until it lands on a word that has neighbours (not every word is
// embedded yet).
async function pickStartWord() {
  for (let i = 0; i < 4; i++) {
    const res = await fetch('/api/browse/words?random=true')
    const word = res.ok ? (await res.json())?.items?.[0] : null
    if (!word) continue
    const nb = await fetch(`/api/words/${word.id}/neighbors?signal=definition&k=1`)
    if (nb.ok && (await nb.json())?.neighbors?.length) return word.id
  }
  return null
}

function VizWordGraph() {
  const navigate = useNavigate()
  // The start word lives in the URL, so a reload or a shared link shows the
  // same graph; only a bare visit picks a new one.
  const [searchParams, setSearchParams] = useSearchParams()
  const wordId = searchParams.get('word')

  useEffect(() => {
    if (wordId) return undefined
    let cancelled = false
    pickStartWord()
      .then((id) => !cancelled && id && setSearchParams({ word: String(id) }, { replace: true }))
      .catch(() => {})
    return () => { cancelled = true }
  }, [wordId, setSearchParams])

  return (
    <VizPage>
      <VizPageHeader title="Words by meaning">
        A word and its nearest neighbours &mdash; by what its definition means, or by how it&rsquo;s spelled. It
        starts on a random word; search for any other. Click a neighbour to open its page.
      </VizPageHeader>
      <div className="viz-graph-embed">
        <Suspense fallback={<div className="page-loading">Loading…</div>}>
          <GraphView initialWordId={wordId ? Number(wordId) : undefined}
            onNodeNavigate={(node) => navigate(`/app/words/${node.id}`)} />
        </Suspense>
      </div>
    </VizPage>
  )
}

export default VizWordGraph
