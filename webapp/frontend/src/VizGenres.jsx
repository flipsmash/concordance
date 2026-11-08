import { Suspense, lazy } from 'react'
import { VizPage, VizPageHeader } from './VizPage'
import './Visualizations.css'

// Lazy: pulls in react-force-graph's canvas/d3-force bundle only here.
const GenreOverlapGraph = lazy(() => import('./GenreOverlapGraph'))

function VizGenres() {
  return (
    <VizPage>
      <VizPageHeader title="Genres by shared vocabulary">
        Every genre, linked to the genres whose books share the most rare words with it. Circle size is how many
        words a genre has; line thickness is how much two genres share. Click a genre&rsquo;s name to focus on it,
        its circle to browse its books.
      </VizPageHeader>
      <div className="viz-graph-embed">
        <Suspense fallback={<div className="page-loading">Loading…</div>}>
          <GenreOverlapGraph />
        </Suspense>
      </div>
    </VizPage>
  )
}

export default VizGenres
