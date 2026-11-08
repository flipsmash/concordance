import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.jsx'

// A deploy can still leave an open tab asking for a chunk that is gone (or
// one the network dropped): reload once to pick up the current build, and
// never loop -- if it fails again right after a reload, leave it be.
window.addEventListener('vite:preloadError', (event) => {
  try {
    const last = Number(sessionStorage.getItem('chunk-reload-at') || 0)
    if (Date.now() - last < 30000) return
    sessionStorage.setItem('chunk-reload-at', String(Date.now()))
  } catch {
    // storage unavailable: still reload once
  }
  event.preventDefault()
  window.location.reload()
})

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
