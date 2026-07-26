import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { useLocation } from 'react-router-dom'

// Static per-section guide words, matched by path prefix. Browse is the only
// section with real dynamic data (the actual first/last lemma on screen) --
// everywhere else this is the whole answer.
const SECTION_DEFAULTS = [
  { test: (p) => p === '/app', left: 'a', center: 'browse', right: 'z' },
  { test: (p) => p.startsWith('/app/authors'), left: 'authors', center: 'writers', right: 'works' },
  { test: (p) => p.startsWith('/app/quiz'), left: 'configure', center: 'quiz', right: 'begin' },
  { test: (p) => p.startsWith('/app/progress'), left: 'sessions', center: 'progress', right: 'streaks' },
  { test: (p) => p.startsWith('/app/sets'), left: 'collections', center: 'sets', right: 'flashcards' },
  { test: (p) => p.startsWith('/app/admin'), left: 'curate', center: 'admin', right: 'review' },
  { test: (p) => p.startsWith('/app/words'), left: 'entry', center: 'word', right: 'detail' },
  { test: (p) => p.startsWith('/app/visualizations'), left: 'similar', center: 'visualize', right: 'related' },
]
const FALLBACK = { left: '', center: '', right: '' }

function defaultsFor(pathname) {
  const match = SECTION_DEFAULTS.find((s) => s.test(pathname))
  return match || FALLBACK
}

const GuideWordContext = createContext(null)

/** Wraps the whole shell. Holds an optional override a leaf page can push up
 * via useGuideWords(). The override is keyed to the pathname that set it and
 * only used while that pathname is still current -- NOT reset by a provider-
 * level effect on route change, because child effects fire before parent
 * effects in React: a naive "reset on location change" in this provider
 * would run after Browse's own effect on the very navigation that lands on
 * Browse, wiping what it just set. Path-keying makes a stale override
 * self-invalidate the instant the route changes, with no ordering
 * dependency and no flash of wrong content. */
export function GuideWordProvider({ children }) {
  const [override, setOverride] = useState(null)
  const location = useLocation()

  const setGuideWords = useCallback((pathname, left, center, right) => {
    setOverride({ pathname, left, center, right })
  }, [])

  const value = useMemo(() => {
    if (override && override.pathname === location.pathname) return override
    return defaultsFor(location.pathname)
  }, [override, location.pathname])

  const ctxValue = useMemo(() => ({ ...value, setGuideWords }), [value, setGuideWords])

  return <GuideWordContext.Provider value={ctxValue}>{children}</GuideWordContext.Provider>
}

/** Read-only -- consumed by AppShell's guide-header bar. */
export function useGuideHeader() {
  const ctx = useContext(GuideWordContext)
  if (!ctx) throw new Error('useGuideHeader must be used within a GuideWordProvider')
  const { left, center, right } = ctx
  return { left, center, right }
}

/** Called by a leaf page (currently only Browse) to push real dynamic text
 * up into the shell's header. */
export function useGuideWords(left, center, right) {
  const ctx = useContext(GuideWordContext)
  if (!ctx) throw new Error('useGuideWords must be used within a GuideWordProvider')
  const location = useLocation()
  const { setGuideWords } = ctx
  useEffect(() => {
    setGuideWords(location.pathname, left, center, right)
  }, [setGuideWords, location.pathname, left, center, right])
}
