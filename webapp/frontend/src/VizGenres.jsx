import { Suspense, lazy } from 'react'
import { VizPage, VizPageHeader } from './VizPage'
import './Visualizations.css'

// Lazy: pulls in react-force-graph's canvas/d3-force bundle only here.
const GenreOverlapGraph = lazy(() => import('./GenreOverlapGraph'))

function VizGenres() {
  return (
    <VizPage>
      <VizPageHeader title="Genres by shared vocabulary">
        Genres linked to the ones whose books share the most rare words with them. <strong>Families</strong> groups
        related genres (speculative, crime &amp; suspense, life writing…); <strong>Genres</strong> shows the
        specific ones, leaving out the catch-all &ldquo;Fiction&rdquo; and &ldquo;Nonfiction&rdquo; tags. Audience
        and form tags &mdash; Children&rsquo;s, Classics, Poetry &mdash; cut across every genre, so they&rsquo;re off
        unless you add them. Circle size is how many words; line thickness is how much two share. Click a name to
        focus on it, its circle to browse its books.
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
