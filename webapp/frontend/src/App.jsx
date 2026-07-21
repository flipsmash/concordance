import { Suspense, lazy } from 'react'
import { NavLink, Outlet, Route, Routes } from 'react-router-dom'
import AcceptedView from './AcceptedView'
import AdminSettings from './AdminSettings'
import './App.css'
import { AuthProvider } from './AuthContext'
import Login from './Login'
import Register from './Register'
import RejectedView from './RejectedView'
import RequireAuth from './RequireAuth'

// Lazy-loaded: pulls in react-force-graph-2d's canvas/d3-force bundle only
// when the Graph tab is actually opened, so Accepted/Rejected stay unaffected.
const GraphView = lazy(() => import('./GraphView'))
const WordDetail = lazy(() => import('./WordDetail'))
const Browse = lazy(() => import('./Browse'))
const QuizConfig = lazy(() => import('./QuizConfig'))
const QuizRun = lazy(() => import('./QuizRun'))
const QuizReview = lazy(() => import('./QuizReview'))
const Authors = lazy(() => import('./Authors'))
const AuthorWorks = lazy(() => import('./AuthorWorks'))
const WorkDetail = lazy(() => import('./WorkDetail'))

function tabClass({ isActive }) {
  return isActive ? 'tab active' : 'tab'
}

function Layout() {
  return (
    <div className="review-app">
      <header>
        <h1>Vocab Review</h1>
        <div className="tabs">
          <NavLink to="/accepted" className={tabClass}>
            Accepted
          </NavLink>
          <NavLink to="/rejected" className={tabClass}>
            Rejected
          </NavLink>
          <NavLink to="/graph" className={tabClass}>
            Graph
          </NavLink>
          <NavLink to="/settings" className={tabClass}>
            Settings
          </NavLink>
        </div>
      </header>

      <Outlet />
    </div>
  )
}

function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<AcceptedView />} />
          <Route path="accepted" element={<AcceptedView />} />
          <Route path="rejected" element={<RejectedView />} />
          <Route path="settings" element={<AdminSettings />} />
          <Route
            path="graph"
            element={
              <Suspense fallback={<div className="page-loading">Loading…</div>}>
                <GraphView />
              </Suspense>
            }
          />
        </Route>
        {/* Outside Layout's 820px-max-width column -- the detail page's embedded
            graph wants more horizontal room than the table views do. */}
        <Route
          path="words/:id"
          element={
            <Suspense fallback={<div className="page-loading">Loading…</div>}>
              <WordDetail />
            </Suspense>
          }
        />

        {/* Non-admin side: app-login instead of Cloudflare Access. Nothing
            under here should ever be added to the Cloudflare Access policy --
            see the user-management plan's "which paths go behind Cloudflare
            Access" note. */}
        <Route path="login" element={<Login />} />
        <Route path="register" element={<Register />} />
        <Route path="app" element={<RequireAuth />}>
          <Route
            index
            element={
              <Suspense fallback={<div className="page-loading">Loading…</div>}>
                <Browse />
              </Suspense>
            }
          />
          <Route
            path="words/:id"
            element={
              <Suspense fallback={<div className="page-loading">Loading…</div>}>
                <WordDetail backTo="/app" />
              </Suspense>
            }
          />
          <Route
            path="quiz"
            element={
              <Suspense fallback={<div className="page-loading">Loading…</div>}>
                <QuizConfig />
              </Suspense>
            }
          />
          <Route
            path="quiz/:sessionId"
            element={
              <Suspense fallback={<div className="page-loading">Loading…</div>}>
                <QuizRun />
              </Suspense>
            }
          />
          <Route
            path="quiz/:sessionId/review"
            element={
              <Suspense fallback={<div className="page-loading">Loading…</div>}>
                <QuizReview />
              </Suspense>
            }
          />
          <Route
            path="authors"
            element={
              <Suspense fallback={<div className="page-loading">Loading…</div>}>
                <Authors />
              </Suspense>
            }
          />
          <Route
            path="authors/:author"
            element={
              <Suspense fallback={<div className="page-loading">Loading…</div>}>
                <AuthorWorks />
              </Suspense>
            }
          />
          <Route
            path="authors/:author/:bookId"
            element={
              <Suspense fallback={<div className="page-loading">Loading…</div>}>
                <WorkDetail />
              </Suspense>
            }
          />
        </Route>
      </Routes>
    </AuthProvider>
  )
}

export default App
