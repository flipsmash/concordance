import { Suspense, lazy } from 'react'
import { useNavigate } from 'react-router-dom'
import { VizPage, VizPageHeader } from './VizPage'
import './Visualizations.css'

const GraphView = lazy(() => import('./GraphView'))

function VizWordGraph() {
  const navigate = useNavigate()
  return (
    <VizPage>
      <VizPageHeader title="Words by meaning">
        Search for a word to see its nearest neighbours &mdash; by what its definition means, or by how it&rsquo;s
        spelled. Click a neighbour to open its page.
      </VizPageHeader>
      <div className="viz-graph-embed">
        <Suspense fallback={<div className="page-loading">Loading…</div>}>
          <GraphView onNodeNavigate={(node) => navigate(`/app/words/${node.id}`)} />
        </Suspense>
      </div>
    </VizPage>
  )
}

export default VizWordGraph
