import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

const API_BASE = ''

// The Categories drilldown's genre filter: one genre, kept in the URL as
// ?genre= so it survives every click deeper (each page passes it through
// withGenre on its links) and a filtered page can be bookmarked/shared.
export function useCategoryGenre() {
  const [params, setParams] = useSearchParams()
  const genre = params.get('genre') || ''
  function setGenre(next) {
    const p = new URLSearchParams(params)
    if (next) p.set('genre', next)
    else p.delete('genre')
    setParams(p, { replace: true })
  }
  return [genre, setGenre]
}

// `path` with the genre appended as `key` -- 'genre' for category pages and
// API calls, 'all_genre' for /app/words (its own genre filter).
export function withGenre(path, genre, key = 'genre') {
  if (!genre) return path
  return `${path}${path.includes('?') ? '&' : '?'}${key}=${encodeURIComponent(genre)}`
}

export function CategoryGenreFilter({ genre, onChange }) {
  const [genres, setGenres] = useState([])

  useEffect(() => {
    fetch(`${API_BASE}/api/browse/genres`)
      .then((res) => (res.ok ? res.json() : []))
      .then(setGenres)
      .catch(() => setGenres([]))
  }, [])

  return (
    <label className="category-genre-filter">
      <span className="muted">Books</span>
      <select value={genre} onChange={(e) => onChange(e.target.value)}>
        <option value="">All books</option>
        {genres.map((g) => <option key={g} value={g}>{g}</option>)}
      </select>
    </label>
  )
}
