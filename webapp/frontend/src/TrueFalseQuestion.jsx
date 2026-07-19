import { useState } from 'react'

function buttonClass(value, chosen, result) {
  if (!result) return chosen === value ? 'tf-option selected' : 'tf-option'
  const isCorrect = result.correct_answer === value
  const isChosen = chosen === value
  if (isCorrect) return 'tf-option reveal-correct'
  if (isChosen) return 'tf-option reveal-incorrect'
  return 'tf-option'
}

/** True/false: a single statement -- "WORD means DEFINITION" -- that's either
 * the word's own definition or a foil's, judged true or false. */
function TrueFalseQuestion({ question, result, disabled, onAnswer }) {
  const [chosen, setChosen] = useState(null)

  function handleClick(value) {
    setChosen(value)
    onAnswer(value)
  }

  return (
    <div className="mc-question">
      <p className="mc-prompt">
        <strong>{question.statement_word}</strong> means: {question.statement_definition}
      </p>
      <div className="tf-options">
        <button type="button" className={buttonClass(true, chosen, result)} disabled={disabled}
          onClick={() => handleClick(true)}>
          True
        </button>
        <button type="button" className={buttonClass(false, chosen, result)} disabled={disabled}
          onClick={() => handleClick(false)}>
          False
        </button>
      </div>
      {result && (
        <p className={result.is_correct ? 'mc-feedback correct' : 'mc-feedback incorrect'}>
          {result.is_correct ? 'Correct.' : `Not quite -- that statement was ${result.correct_answer ? 'true' : 'false'}.`}
        </p>
      )}
    </div>
  )
}

export default TrueFalseQuestion
