import { useState } from 'react'

function optionClass(option, selected, result) {
  if (!result) return selected === option.word_id ? 'mc-option selected' : 'mc-option'
  const isCorrect = option.word_id === result.correct_word_id
  const isChosen = option.word_id === selected
  if (isCorrect) return 'mc-option reveal-correct'
  if (isChosen) return 'mc-option reveal-incorrect'
  return 'mc-option'
}

/** One multiple-choice question: a prompt (definition or word, depending on the
 * session's direction) and its shuffled options. `result` is only set once the
 * session's feedback_timing is 'immediate' and this question has been answered --
 * under 'end_of_test' the parent advances straight to the next question instead. */
function McQuestion({ question, result, disabled, onSelect }) {
  const [selected, setSelected] = useState(null)

  function handleClick(wordId) {
    setSelected(wordId)
    onSelect(wordId)
  }

  return (
    <div className="mc-question">
      <p className="mc-prompt">{question.prompt}</p>
      <div className="mc-options">
        {question.options.map((opt) => (
          <button
            type="button"
            key={opt.word_id ?? 'nota'}
            className={optionClass(opt, selected, result)}
            disabled={disabled}
            onClick={() => handleClick(opt.word_id)}
          >
            {opt.label}
          </button>
        ))}
      </div>
      {result && (
        <p className={result.is_correct ? 'mc-feedback correct' : 'mc-feedback incorrect'}>
          {result.is_correct ? 'Correct.' : `Not quite -- the answer was "${result.correct_label}".`}
        </p>
      )}
    </div>
  )
}

export default McQuestion
