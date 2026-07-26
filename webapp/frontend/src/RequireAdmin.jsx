import { Navigate, Outlet } from 'react-router-dom'
import { useAuth } from './AuthContext'

// Nested INSIDE the already-RequireAuth-gated /app tree, so `user` is
// guaranteed non-null by the time this runs -- a stricter check layered on
// top of "logged in," not a variant of it, hence its own file rather than
// an `adminOnly` prop on RequireAuth: a non-admin who's already logged in
// isn't unauthenticated, so it redirects to /app, never to /login.
function RequireAdmin() {
  const { user } = useAuth()
  if (!user?.is_admin) return <Navigate to="/app" replace />
  return <Outlet />
}

export default RequireAdmin
