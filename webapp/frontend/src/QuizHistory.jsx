import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import './QuizHistory.css'

const API_BASE = ''
const WEEKDAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

// Same score->color convention QuizReview's hero ring already uses (high/mid/low
// at 80/50), so a day's dot color means the same thing there and here.
function scoreClass(pct) {
  if (pct >= 80) return 'quiz-history-dot high'
  if (pct >= 50) return 'quiz-history-dot mid'
  return 'quiz-history-dot low'
}

// UTC day-key, matching progress.py's own "streak buckets by UTC day"
// convention -- a session finished at 11pm US-Eastern shouldn't land on a
// different calendar day here than it does in the practice-streak tile.
// finished_at comes back with a real UTC offset (e.g. -04:00), so this must
// parse it into an actual Date and re-derive the UTC date -- slicing the raw
// string would silently use the offset's *local* wall-clock date instead.
function dayKey(iso) {
  return new Date(iso).toISOString().slice(0, 10)
}

function firstOfMonthUTC(year, month) {
  return new Date(Date.UTC(year, month, 1))
}

function QuizHistory() {
  const navigate = useNavigate()
  const [history, setHistory] = useState(null)
  const [error, setError] = useState('')
  const [viewedMonth, setViewedMonth] = useState(() => {
    const now = new Date()
    return firstOfMonthUTC(now.getUTCFullYear(), now.getUTCMonth())
  })
  const [expandedDay, setExpandedDay] = useState(null) // day-key string | null

  useEffect(() => {
    fetch(`${API_BASE}/api/progress/history`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to load quiz history'))))
      .then(setHistory)
      .catch((err) => setError(err.message))
  }, [])

  const byDay = useMemo(() => {
    const map = new Map()
    for (const s of history ?? []) {
      const key = dayKey(s.finished_at)
      if (!map.has(key)) map.set(key, [])
      map.get(key).push(s)
    }
    for (const sessions of map.values()) {
      sessions.sort((a, b) => a.finished_at.localeCompare(b.finished_at))
    }
    return map
  }, [history])

  const now = new Date()
  const isCurrentMonth =
    viewedMonth.getUTCFullYear() === now.getUTCFullYear() && viewedMonth.getUTCMonth() === now.getUTCMonth()

  function shiftMonth(delta) {
    setExpandedDay(null)
    setViewedMonth((m) => firstOfMonthUTC(m.getUTCFullYear(), m.getUTCMonth() + delta))
  }

  function handleDayClick(key, sessions) {
    if (!sessions || sessions.length === 0) return
    if (sessions.length === 1) {
      navigate(`/app/quiz/${sessions[0].session_id}/review`)
      return
    }
    setExpandedDay((d) => (d === key ? null : key))
  }

  const year = viewedMonth.getUTCFullYear()
  const month = viewedMonth.getUTCMonth()
  const daysInMonth = new Date(Date.UTC(year, month + 1, 0)).getUTCDate()
  const startWeekday = firstOfMonthUTC(year, month).getUTCDay()
  const monthLabel = viewedMonth.toLocaleDateString(undefined, { month: 'long', year: 'numeric', timeZone: 'UTC' })

  const cells = []
  for (let i = 0; i < startWeekday; i++) cells.push(null)
  for (let d = 1; d <= daysInMonth; d++) cells.push(d)

  if (error) return <div className="error-banner">{error}</div>
  if (!history) return <div className="page-loading">Loading…</div>

  return (
    <div className="quiz-history-page">
      <header className="quiz-history-header">
        <h1>Quiz history</h1>
      </header>

      {history.length === 0 ? (
        <p className="progress-empty-state">No finished quizzes yet -- take a quiz to start building your history.</p>
      ) : (
        <>
          <div className="quiz-history-nav">
            <button type="button" onClick={() => shiftMonth(-1)}>← Prev</button>
            <span className="quiz-history-month">{monthLabel}</span>
            <button type="button" onClick={() => shiftMonth(1)} disabled={isCurrentMonth}>Next →</button>
          </div>

          <div className="quiz-history-grid">
            {WEEKDAY_LABELS.map((w) => (
              <div key={w} className="quiz-history-weekday">{w}</div>
            ))}
            {cells.map((d, i) => {
              if (d === null) return <div key={`blank-${i}`} className="quiz-history-cell empty" />
              const key = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`
              const sessions = byDay.get(key)
              return (
                <div
                  key={key}
                  className={sessions ? 'quiz-history-cell has-sessions' : 'quiz-history-cell'}
                  onClick={() => handleDayClick(key, sessions)}
                >
                  <span className="quiz-history-daynum">{d}</span>
                  {sessions && (
                    <span className="quiz-history-dots">
                      {sessions.slice(0, 4).map((s) => (
                        <span key={s.session_id} className={scoreClass(s.score_pct)} title={`${Math.round(s.score_pct)}%`} />
                      ))}
                      {sessions.length > 4 && <span className="quiz-history-more">+{sessions.length - 4}</span>}
                    </span>
                  )}
                </div>
              )
            })}
          </div>

          {expandedDay && byDay.get(expandedDay) && (
            <div className="quiz-history-day-detail">
              <h2>
                {new Date(`${expandedDay}T00:00:00Z`).toLocaleDateString(undefined, {
                  weekday: 'long', month: 'long', day: 'numeric', timeZone: 'UTC',
                })}
              </h2>
              <ul className="quiz-history-day-list">
                {byDay.get(expandedDay).map((s) => (
                  <li key={s.session_id}>
                    <Link to={`/app/quiz/${s.session_id}/review`}>
                      <span className={scoreClass(s.score_pct)} />
                      {new Date(s.finished_at).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })}
                      {' · '}
                      {Math.round(s.score_pct)}% ({s.total_questions} question{s.total_questions === 1 ? '' : 's'})
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default QuizHistory
