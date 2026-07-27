import { NavLink, Outlet } from 'react-router-dom'
import './AppShell.css'
import { useAuth } from './AuthContext'
import { GuideWordProvider, useGuideHeader } from './GuideWordContext'

// The destinations a reader can always reach, one click away, from anywhere
// in the app -- the whole point of this shell. Admin is filtered out
// entirely (not just hidden) for a non-admin user, so the "curation chrome
// visible to anyone" gap this shell replaces doesn't reappear here. Letter
// is normally the label's own first letter (the thumb-index conceit); where
// that collides with an earlier tab (Books vs. Browse) or isn't a letter at
// all (Admin), a distinct glyph is used instead -- never a wrong letter.
const SECTIONS = [
  { key: 'browse', letter: 'B', label: 'Browse', to: '/app', end: true },
  { key: 'books', letter: '❦', label: 'Books', to: '/app/books' },
  { key: 'authors', letter: 'A', label: 'Authors', to: '/app/authors' },
  { key: 'categories', letter: 'C', label: 'Categories', to: '/app/categories' },
  { key: 'visualizations', letter: 'V', label: 'Visuals', to: '/app/visualizations' },
  { key: 'quiz', letter: 'Q', label: 'Quiz', to: '/app/quiz' },
  { key: 'progress', letter: 'P', label: 'Progress', to: '/app/progress' },
  { key: 'sets', letter: 'S', label: 'Sets', to: '/app/sets' },
  { key: 'admin', letter: '✦', label: 'Admin', to: '/app/admin', adminOnly: true },
]

function tabClass({ isActive }) {
  return isActive ? 'thumb-rail-tab active' : 'thumb-rail-tab'
}

function ThumbRail() {
  const { user, logout } = useAuth()
  const visible = SECTIONS.filter((s) => !s.adminOnly || user?.is_admin)

  return (
    <nav className="thumb-rail" aria-label="Main sections">
      {visible.map((s) => (
        <NavLink
          key={s.key}
          to={s.to}
          end={s.end}
          className={s.adminOnly ? ({ isActive }) => `${tabClass({ isActive })} admin-tab` : tabClass}
        >
          <span className="thumb-rail-letter" aria-hidden="true">{s.letter}</span>
          <span className="thumb-rail-label">{s.label}</span>
        </NavLink>
      ))}
      <div className="rail-divider" />
      <div className="rail-foot">
        <span className="rail-foot-user">{user?.username}</span>
        <button type="button" className="rail-foot-logout" onClick={logout}>Log out</button>
      </div>
    </nav>
  )
}

function GuideHeaderBar() {
  const { left, center, right } = useGuideHeader()
  return (
    <div className="guide-header">
      <span className="guide-word guide-word-left">{left}</span>
      <span className="guide-label">{center}</span>
      <span className="guide-word guide-word-right">{right}</span>
    </div>
  )
}

/** Persistent shell for every page under /app -- a thumb-index tab rail
 * (styled after the die-cut alphabet tabs on a dictionary's fore-edge) as
 * the one, single, always-visible way to reach every section a reader is
 * allowed into, plus a guide-word header (a dictionary spread's running
 * head) repurposed as live wayfinding. Existing pages keep their own
 * internal layout/content unchanged -- this only supplies the chrome
 * around them. */
function AppShell() {
  return (
    <GuideWordProvider>
      <div className="app-shell">
        <ThumbRail />
        <div className="app-shell-content">
          <GuideHeaderBar />
          <div className="app-shell-outlet">
            <Outlet />
          </div>
        </div>
      </div>
    </GuideWordProvider>
  )
}

export default AppShell
