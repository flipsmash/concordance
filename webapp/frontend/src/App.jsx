import { useState } from 'react'
import './App.css'
import AcceptedView from './AcceptedView'
import RejectedView from './RejectedView'

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
        </div>
      </header>

      {tab === 'accepted' ? <AcceptedView /> : <RejectedView />}
    </div>
  )
}

export default App
