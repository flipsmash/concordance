import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from './AuthContext'
import './Browse.css'

const API_BASE = ''

// Search entry point for logged-in non-admin browsing. Debounced
// search-as-you-type against the same /api/words/search endpoint the admin
// Graph tab's search box uses (GraphView.jsx) -- same pattern, not shared as
// a component since GraphView's version is entangled with its own
// suggestion-dropdown refs/effects.
function Browse() {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const searchRef = useRef(null)
  const navigate = useNavigate()
  const { user, logout } = useAuth()

  useEffect(() => {
    function handlePointerDown(e) {
      if (searchRef.current && !searchRef.current.contains(e.target)) setSuggestions([])
    }
    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [])

  useEffect(() => {
    if (!query.trim()) {
      setSuggestions([])
      return
    }
    const handle = setTimeout(() => {
      fetch(`${API_BASE}/api/words/search?q=${encodeURIComponent(query)}&limit=15`)
        .then((res) => res.json())
        .then(setSuggestions)
        .catch(() => {})
    }, 200)
    return () => clearTimeout(handle)
  }, [query])

  return (
    <div className="browse-page">
      <header className="browse-header">
        <h1>Vocab Browse</h1>
        <div className="browse-user">
          {user?.username} · <button type="button" onClick={logout}>Log out</button>
        </div>
      </header>

      <div className="browse-search" ref={searchRef}>
        <input
          type="text"
          placeholder="Search for a word…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          autoFocus
        />
        {suggestions.length > 0 && (
          <ul className="browse-suggestions">
            {suggestions.map((w) => (
              <li key={w.id} onClick={() => navigate(`/app/words/${w.id}`)}>
                {w.lemma}
              </li>
            ))}
          </ul>
        )}
      </div>

      {!query.trim() && <p className="browse-hint">Search for a word to see its full profile.</p>}
    </div>
  )
}

export default Browse
