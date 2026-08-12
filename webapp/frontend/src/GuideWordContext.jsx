import { useLocation } from 'react-router-dom'

// Per-section label used only for the shell's central brand mark's
// aria-label (e.g. "concordance -- browse") -- see AppShell.jsx's
// GuideHeaderBar. The header's own left/right slots used to show a
// similar per-section word pair (mostly static, e.g. "authors"/"works";
// real dynamic first/last-on-screen text only on Browse) until they were
// replaced by the always-on word/definition search boxes, which removed
// the only reason this used to be a context with a leaf-page override
// mechanism (Browse pushing its own live text up into the shell) --
// nothing overrides `center` dynamically, so this is a plain per-path
// lookup now, no provider needed.
const CENTER_LABELS = [
  { test: (p) => p === '/app', center: 'home' },
  { test: (p) => p === '/app/words', center: 'browse' },
  { test: (p) => p.startsWith('/app/books'), center: 'books' },
  { test: (p) => p.startsWith('/app/authors'), center: 'writers' },
  { test: (p) => p.startsWith('/app/quiz/history'), center: 'quiz history' },
  { test: (p) => p.startsWith('/app/quiz'), center: 'quiz' },
  { test: (p) => p.startsWith('/app/progress'), center: 'progress' },
  { test: (p) => p.startsWith('/app/sets'), center: 'sets' },
  { test: (p) => p.startsWith('/app/admin'), center: 'admin' },
  { test: (p) => p.startsWith('/app/words'), center: 'word' },
  { test: (p) => p.startsWith('/app/visualizations'), center: 'visualize' },
]

export function useGuideCenter() {
  const { pathname } = useLocation()
  return (CENTER_LABELS.find((s) => s.test(pathname)) || { center: '' }).center
}
