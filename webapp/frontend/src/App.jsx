import { Suspense, lazy } from 'react'
import { Navigate, NavLink, Outlet, Route, Routes } from 'react-router-dom'
import AcceptedView from './AcceptedView'
import AdminSettings from './AdminSettings'
import AppShell from './AppShell'
import './App.css'
import { AuthProvider } from './AuthContext'
import Login from './Login'
import Register from './Register'
import RejectedView from './RejectedView'
import RequestInvite from './RequestInvite'
import RequireAdmin from './RequireAdmin'
import RequireAuth from './RequireAuth'

// Lazy-loaded: pulls in react-force-graph-2d's canvas/d3-force bundle only
// when the Graph tab is actually opened, so Accepted/Rejected stay unaffected.
const GraphView = lazy(() => import('./GraphView'))
const WordDetail = lazy(() => import('./WordDetail'))
const Home = lazy(() => import('./Home'))
const Browse = lazy(() => import('./Browse'))
const QuizConfig = lazy(() => import('./QuizConfig'))
const QuizRun = lazy(() => import('./QuizRun'))
const QuizReview = lazy(() => import('./QuizReview'))
const QuizHistory = lazy(() => import('./QuizHistory'))
const Authors = lazy(() => import('./Authors'))
const AuthorWorks = lazy(() => import('./AuthorWorks'))
const Books = lazy(() => import('./Books'))
const WorkDetail = lazy(() => import('./WorkDetail'))
const BookRelatedness = lazy(() => import('./BookRelatedness'))
const AuthorRelatedness = lazy(() => import('./AuthorRelatedness'))
const AuthorsRelatedness = lazy(() => import('./AuthorsRelatedness'))
const BooksRelatedness = lazy(() => import('./BooksRelatedness'))
const CategoriesOverview = lazy(() => import('./CategoriesOverview'))
const CategoryBucketDetail = lazy(() => import('./CategoryBucketDetail'))
const CategoryFieldDetail = lazy(() => import('./CategoryFieldDetail'))
const CategorySubfieldDetail = lazy(() => import('./CategorySubfieldDetail'))
const CategorySubsubfieldDetail = lazy(() => import('./CategorySubsubfieldDetail'))
const CategorySubsubsubfieldDetail = lazy(() => import('./CategorySubsubsubfieldDetail'))
const Visualizations = lazy(() => import('./Visualizations'))
const DomainMap = lazy(() => import('./DomainMap'))
const Sets = lazy(() => import('./Sets'))
const SetDetail = lazy(() => import('./SetDetail'))
const FlashcardRun = lazy(() => import('./FlashcardRun'))
const Progress = lazy(() => import('./Progress'))
const OedEntries = lazy(() => import('./OedEntries'))
const OedEntryDetail = lazy(() => import('./OedEntryDetail'))

function tabClass({ isActive }) {
  return isActive ? 'tab active' : 'tab'
}

// The admin curation shell -- now nested under /app/admin, inside AppShell's
// thumb rail (reached via its own "Admin" tab, role-gated by RequireAdmin
// one level up) rather than living at the site root with no way back to the
// reader side. Tab targets are relative (accepted/rejected/...), which is
// what makes this component correct regardless of the /app/admin prefix.
function Layout() {
  return (
    <div className="review-app">
      <header>
        <h1>Vocab Review</h1>
        <div className="tabs">
          <NavLink to="accepted" className={tabClass}>
            Accepted
          </NavLink>
          <NavLink to="rejected" className={tabClass}>
            Rejected
          </NavLink>
          <NavLink to="graph" className={tabClass}>
            Graph
          </NavLink>
          <NavLink to="settings" className={tabClass}>
            Settings
          </NavLink>
          <NavLink to="oed" className={tabClass}>
            OED
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
        <Route index element={<Navigate to="/app" replace />} />

        {/* Outside AppShell's rail entirely -- the detail page's embedded
            graph wants more horizontal room than a page inside the shell
            has, and this mount has no persistent nav around it, so it keeps
            its own "back" link (see WordDetail's showBackLink prop). */}
        <Route
          path="words/:id"
          element={
            <Suspense fallback={<div className="page-loading">Loading…</div>}>
              <WordDetail backTo="/app/admin/accepted" />
            </Suspense>
          }
        />

        {/* Non-admin side: app-login instead of Cloudflare Access. Nothing
            under here should ever be added to the Cloudflare Access policy --
            see the user-management plan's "which paths go behind Cloudflare
            Access" note. */}
        <Route path="login" element={<Login />} />
        <Route path="register" element={<Register />} />
        <Route path="request-invite" element={<RequestInvite />} />
        <Route path="app" element={<RequireAuth />}>
          <Route element={<AppShell />}>
            <Route
              index
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <Home />
                </Suspense>
              }
            />
            <Route
              path="words"
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
                  <WordDetail backTo="/app/words" showBackLink={false} />
                </Suspense>
              }
            />
            <Route
              path="visualizations"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <Visualizations />
                </Suspense>
              }
            />
            <Route
              path="visualizations/domain-map"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <DomainMap />
                </Suspense>
              }
            />
            <Route
              path="progress"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <Progress />
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
              path="quiz/history"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <QuizHistory />
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
              path="sets"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <Sets />
                </Suspense>
              }
            />
            <Route
              path="sets/:setId"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <SetDetail />
                </Suspense>
              }
            />
            <Route
              path="sets/:setId/flashcards"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <FlashcardRun />
                </Suspense>
              }
            />
            <Route
              path="books"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <Books />
                </Suspense>
              }
            />
            <Route
              path="books/relatedness"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <BooksRelatedness />
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
              path="authors/relatedness"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <AuthorsRelatedness />
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
              path="authors/:author/relatedness"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <AuthorRelatedness />
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
            <Route
              path="authors/:author/:bookId/relatedness"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <BookRelatedness />
                </Suspense>
              }
            />
            <Route
              path="categories"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <CategoriesOverview />
                </Suspense>
              }
            />
            <Route
              path="categories/:bucket"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <CategoryBucketDetail />
                </Suspense>
              }
            />
            <Route
              path="categories/:bucket/:code"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <CategoryFieldDetail />
                </Suspense>
              }
            />
            <Route
              path="categories/:bucket/:code/:subcode"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <CategorySubfieldDetail />
                </Suspense>
              }
            />
            <Route
              path="categories/:bucket/:code/:subcode/:subsubcode"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <CategorySubsubfieldDetail />
                </Suspense>
              }
            />
            <Route
              path="categories/:bucket/:code/:subcode/:subsubcode/:subsubsubcode"
              element={
                <Suspense fallback={<div className="page-loading">Loading…</div>}>
                  <CategorySubsubsubfieldDetail />
                </Suspense>
              }
            />

            {/* Admin, one level deeper, gated additionally by is_admin --
                closes the gap where this chrome used to render (with every
                data fetch then 401/403ing) for anyone, admin or not. */}
            <Route path="admin" element={<RequireAdmin />}>
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
                <Route
                  path="oed"
                  element={
                    <Suspense fallback={<div className="page-loading">Loading…</div>}>
                      <OedEntries />
                    </Suspense>
                  }
                />
                <Route
                  path="oed/:entryId"
                  element={
                    <Suspense fallback={<div className="page-loading">Loading…</div>}>
                      <OedEntryDetail />
                    </Suspense>
                  }
                />
              </Route>
            </Route>
          </Route>
        </Route>
      </Routes>
    </AuthProvider>
  )
}

export default App
