import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import './QuizReview.css'

const API_BASE = ''

function scoreClass(pct) {
  if (pct >= 80) return 'quiz-score-ring high'
  if (pct >= 50) return 'quiz-score-ring mid'
  return 'quiz-score-ring low'
}

function itemClass(credit) {
  if (credit >= 1) return 'quiz-review-item correct'
  if (credit <= 0) return 'quiz-review-item incorrect'
  return 'quiz-review-item partial'
}

function itemMark(credit) {
  if (credit >= 1) return '✓'
  if (credit <= 0) return '✕'
  return `${Math.round(credit * 100)}%`
}

function QuizReview() {
  const { sessionId } = useParams()
  const [review, setReview] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/api/quiz/${sessionId}/review`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to load review'))))
      .then(setReview)
      .catch((err) => setError(err.message))
  }, [sessionId])

  if (error) return <div className="error-banner">{error}</div>
  if (!review) return <div className="page-loading">Loading…</div>

  const pct = Math.round(review.score_pct ?? 0)
  const totalCredit = review.items.reduce((sum, i) => sum + i.credit, 0)
  const creditLabel = Number.isInteger(totalCredit) ? totalCredit : totalCredit.toFixed(1)

  return (
    <div className="quiz-review-page">
      <div className="quiz-score-hero">
        <div className={scoreClass(pct)}>
          <span className="quiz-score-number">{pct}</span>
          <span className="quiz-score-percent">%</span>
        </div>
        <p className="quiz-score-caption">
          {creditLabel} of {review.items.length} correct
        </p>
      </div>

      <ul className="quiz-review-list">
        {review.items.map((item) => (
          <li key={item.seq} className={itemClass(item.credit)}>
            <div className="quiz-review-item-head">
              <span className="quiz-review-lemma">{item.target_lemma}</span>
              <span className={`quiz-review-mark ${item.credit >= 1 ? 'correct' : item.credit <= 0 ? 'incorrect' : 'partial'}`}>
                {itemMark(item.credit)}
              </span>
            </div>
            <p className="quiz-review-definition">{item.quiz_definition}</p>
            {item.credit < 1 && (
              <p className="quiz-review-your-answer">
                {item.question_type === 'matching'
                  ? (item.your_label ?? 'No answer')
                  : <>Your answer: {item.your_label ?? <em>no answer</em>}</>}
              </p>
            )}
          </li>
        ))}
      </ul>

      <div className="quiz-review-actions">
        <Link to="/app/quiz" className="quiz-start-btn">
          New quiz
        </Link>
        <Link to="/app" className="quiz-review-back">
          ← Back to browse
        </Link>
      </div>
    </div>
  )
}

export default QuizReview
