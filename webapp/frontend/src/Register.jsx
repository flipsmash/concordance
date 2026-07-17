import { useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from './AuthContext'
import './Auth.css'

const API_BASE = ''

function Register() {
  const [params] = useSearchParams()
  const token = params.get('token') || ''
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const { refresh } = useAuth()
  const navigate = useNavigate()

  function handleSubmit(e) {
    e.preventDefault()
    if (password !== confirm) {
      setError("passwords don't match")
      return
    }
    setSubmitting(true)
    setError('')
    fetch(`${API_BASE}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token, username, password }),
    })
      .then((res) => {
        if (!res.ok) return res.json().then((d) => Promise.reject(new Error(d.detail || 'registration failed')))
        return refresh()
      })
      .then(() => navigate('/app'))
      .catch((err) => setError(err.message))
      .finally(() => setSubmitting(false))
  }

  if (!token) {
    return (
      <div className="auth-page">
        <div className="auth-form">
          <h1>Create account</h1>
          <p className="auth-error">This page needs an invite link — ask the admin for one.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="auth-page">
      <form className="auth-form" onSubmit={handleSubmit}>
        <h1>Create account</h1>
        {error && <div className="auth-error">{error}</div>}
        <label>
          Username
          <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} autoFocus required />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            minLength={8}
            required
          />
        </label>
        <label>
          Confirm password
          <input
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            minLength={8}
            required
          />
        </label>
        <button type="submit" disabled={submitting}>
          {submitting ? 'Creating account…' : 'Create account'}
        </button>
      </form>
    </div>
  )
}

export default Register
