import { useEffect, useState } from 'react'
import './AdminSettings.css'

const API_BASE = ''

// Admin-only: one global toggle today (quiz feedback timing), stored in the
// generic app_settings key/value table so future toggles land here without a
// new page. Reachable only because it's a tab inside the admin curation UI --
// see require_admin in webapp/backend/quiz.py for the backend side.
function AdminSettings() {
  const [settings, setSettings] = useState(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE}/api/admin/settings`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to load settings'))))
      .then((data) => setSettings(data.settings))
      .catch((err) => setError(err.message))
  }, [])

  function setFeedbackMode(mode) {
    setSaving(true)
    setError('')
    fetch(`${API_BASE}/api/admin/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: 'quiz_feedback_timing', value: { mode } }),
    })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to save'))))
      .then((data) => setSettings(data.settings))
      .catch((err) => setError(err.message))
      .finally(() => setSaving(false))
  }

  if (error) return <div className="error-banner">{error}</div>
  if (!settings) return <div className="page-loading">Loading…</div>

  const mode = settings.quiz_feedback_timing?.mode ?? 'immediate'

  return (
    <div className="admin-settings">
      <section className="admin-settings-section">
        <h2>Quiz feedback timing</h2>
        <p className="admin-settings-hint">
          Applies to every quiz-taker going forward. In-progress quizzes keep whatever mode was active when they
          started.
        </p>
        <div className="admin-settings-choice">
          <label>
            <input
              type="radio"
              name="feedback-timing"
              checked={mode === 'immediate'}
              disabled={saving}
              onChange={() => setFeedbackMode('immediate')}
            />
            Immediate -- reveal correct/incorrect after each question
          </label>
          <label>
            <input
              type="radio"
              name="feedback-timing"
              checked={mode === 'end_of_test'}
              disabled={saving}
              onChange={() => setFeedbackMode('end_of_test')}
            />
            End of test -- reveal everything on the review screen only
          </label>
        </div>
      </section>
    </div>
  )
}

export default AdminSettings
