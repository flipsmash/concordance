import { Suspense, lazy } from 'react'
import { VizPage, VizPageHeader } from './VizPage'
import './Visualizations.css'

// Lazy: sigma/WebGL only downloads here.
const ForceMap = lazy(() => import('./ForceMap'))

function VizWordsMap() {
  return (
    <VizPage wide>
      <VizPageHeader title="Every word by meaning">
        The whole vocabulary on one force-directed map: each word pulled toward the words whose definitions mean
        the most alike, so related words gather into neighbourhoods. Zoom in to read them; click a word to open it.
      </VizPageHeader>
      <Suspense fallback={<div className="page-loading">Loading…</div>}>
        <ForceMap entity="word" />
      </Suspense>
    </VizPage>
  )
}

export default VizWordsMap
