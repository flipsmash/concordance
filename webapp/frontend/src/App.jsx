import { Suspense, lazy, useState } from 'react'
import './App.css'
import AcceptedView from './AcceptedView'
import RejectedView from './RejectedView'

// Lazy-loaded: pulls in react-force-graph-2d's canvas/d3-force bundle only
// when the Graph tab is actually opened, so Accepted/Rejected stay unaffected.
const GraphView = lazy(() => import('./GraphView'))

function App() {
  const [tab, setTab] = useState('accepted')

  return (
    <div className="review-app">
      <header>
        <h1>Vocab Review</h1>
        <div className="tabs">
          <button
            type="button"
            className={tab === 'accepted' ? 'tab active' : 'tab'}
            onClick={() => setTab('accepted')}
          >
            Accepted
          </button>
          <button
            type="button"
            className={tab === 'rejected' ? 'tab active' : 'tab'}
            onClick={() => setTab('rejected')}
          >
            Rejected
          </button>
          <button
            type="button"
            className={tab === 'graph' ? 'tab active' : 'tab'}
            onClick={() => setTab('graph')}
          >
            Graph
          </button>
        </div>
      </header>

      {tab === 'accepted' && <AcceptedView />}
      {tab === 'rejected' && <RejectedView />}
      {tab === 'graph' && (
        <Suspense fallback={<div>Loading…</div>}>
          <GraphView />
        </Suspense>
      )}
    </div>
  )
}

export default App
