import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import MatchingQuestion from './MatchingQuestion'
import McQuestion from './McQuestion'
import './QuizRun.css'
import TrueFalseQuestion from './TrueFalseQuestion'

const API_BASE = ''

function QuizRun() {
  const { sessionId } = useParams()
  const navigate = useNavigate()

  const [state, setState] = useState(null) // {total_questions, answered, question, feedback_timing}
  const [error, setError] = useState('')
  const [result, setResult] = useState(null) // immediate-feedback result, or null
  const [submitting, setSubmitting] = useState(false)

  const loadState = useCallback(() => {
    setResult(null)
    return fetch(`${API_BASE}/api/quiz/${sessionId}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to load quiz'))))
      .then((data) => {
        if (data.completed) {
          return fetch(`${API_BASE}/api/quiz/${sessionId}/finish`, { method: 'POST' }).then(() =>
            navigate(`/app/quiz/${sessionId}/review`, { replace: true })
          )
        }
        setState(data)
      })
      .catch((err) => setError(err.message))
  }, [sessionId, navigate])

  useEffect(() => {
    loadState()
  }, [loadState])

  // `extra` merges into the request body -- selected_word_id for mc, answer for
  // true_false, pairs for matching -- each sub-component knows its own shape.
  function submitAnswer(extra) {
    if (submitting || (result && state.feedback_timing === 'immediate')) return
    setSubmitting(true)
    fetch(`${API_BASE}/api/quiz/${sessionId}/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question_id: state.question.question_id, ...extra }),
    })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to submit answer'))))
      .then((data) => {
        if (state.feedback_timing === 'immediate') {
          setResult(data)
          setSubmitting(false)
        } else {
          loadState().finally(() => setSubmitting(false))
        }
      })
      .catch((err) => {
        setError(err.message)
        setSubmitting(false)
      })
  }

  if (error) return <div className="error-banner">{error}</div>
  if (!state || !state.question) return <div className="page-loading">Loading…</div>

  const progress = Math.round((state.answered / state.total_questions) * 100)
  const disabled = submitting || Boolean(result)
  const { question } = state

  return (
    <div className="quiz-run-page">
      <div className="quiz-progress-track">
        <div className="quiz-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <p className="quiz-progress-label">
        Question {state.answered + 1} of {state.total_questions}
      </p>

      {question.question_type === 'mc' && (
        <McQuestion key={question.question_id} question={question} result={result} disabled={disabled}
          onSelect={(wordId) => submitAnswer({ selected_word_id: wordId })} />
      )}
      {question.question_type === 'true_false' && (
        <TrueFalseQuestion key={question.question_id} question={question} result={result} disabled={disabled}
          onAnswer={(answer) => submitAnswer({ answer })} />
      )}
      {question.question_type === 'matching' && (
        <MatchingQuestion key={question.question_id} question={question} result={result} disabled={disabled}
          onSubmit={(pairs) => submitAnswer({ pairs })} />
      )}

      {result && (
        <button type="button" className="quiz-next-btn" onClick={loadState}>
          Next question →
        </button>
      )}
    </div>
  )
}

export default QuizRun
