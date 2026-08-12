import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

const API_BASE = ''

// Lives in the shell's guide-header, left and right of the brand mark (see
// AppShell.jsx) -- replaces the per-section static/guide-word text that
// used to sit there with a live, always-available way to jump straight to
// a word from anywhere in the app. Two instances, same component: mode
// "lemma" searches word.lemma (GET /api/browse/words?q=), mode
// "definition" searches word.definition (GET /api/browse/words?
// definition_q=, see browse_words' own docstring for the word_similarity
// matching that powers it). Both resolve to a WORD either way -- a
// definition match still belongs to exactly one word -- so picking a
// result always lands on that word's detail page, same target regardless
// of which box found it.
//
// Same debounced-fetch/dropdown/outside-click-close shape as AuthorSelect/
// BookSelect, but simpler: no "chosen" chip state, since picking a result
// navigates away immediately rather than setting a filter this component
// itself holds.
function HeaderSearch({ mode, placeholder }) {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    function handlePointerDown(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [])

  useEffect(() => {
    if (!open || !query.trim()) {
      setSuggestions([])
      return
    }
    const handle = setTimeout(() => {
      const params = new URLSearchParams({ page_size: '8' })
      params.set(mode === 'definition' ? 'definition_q' : 'q', query.trim())
      fetch(`${API_BASE}/api/browse/words?${params}`)
        .then((res) => res.json())
        .then((data) => setSuggestions(data.items))
        .catch(() => {})
    }, 200)
    return () => clearTimeout(handle)
  }, [query, open, mode])

  function pick(word) {
    navigate(`/app/words/${word.id}`)
    setQuery('')
    setOpen(false)
  }

  // Definition mode only: the live dropdown above is deliberately a fuzzy,
  // top-8 preview (word_similarity via definition_q) -- good for "is this
  // roughly what I'm thinking of," not for "show me every match." Enter
  // instead lands on the full Words browse list, filtered by literal
  // substring containment (definition_contains -- see browse_words' own
  // comment on why that's a different, non-fuzzy filter from definition_q).
  // Lemma mode has no Enter behavior: the dropdown alone already resolves
  // "find this specific word" well enough that a second, list-everything
  // path isn't a clear win the way it is for a definition phrase.
  function handleKeyDown(e) {
    if (mode !== 'definition' || e.key !== 'Enter' || !query.trim()) return
    navigate(`/app/words?definition_contains=${encodeURIComponent(query.trim())}`)
    setQuery('')
    setOpen(false)
  }

  return (
    <div className={`header-search header-search-box-${mode}`} ref={ref}>
      <input
        type="text"
        className="header-search-input"
        placeholder={placeholder}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKeyDown}
        aria-label={placeholder}
      />
      {open && suggestions.length > 0 && (
        <ul className="header-search-list">
          {suggestions.map((w) => (
            <li key={w.id} onClick={() => pick(w)}>
              <span className="header-search-result-lemma">{w.lemma}</span>
              {mode === 'definition' && w.definition && (
                <span className="header-search-result-definition">{w.definition}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default HeaderSearch
