import { useState } from 'react'

function optionClass(option, selected, result) {
  if (!result) return selected === option.key ? 'mc-option selected' : 'mc-option'
  const isCorrect = option.key === result.correct_option_key
  const isChosen = option.key === selected
  if (isCorrect) return 'mc-option reveal-correct'
  if (isChosen) return 'mc-option reveal-incorrect'
  return 'mc-option'
}

/** One MAT-style analogy question ("A is to B as C is to ___"). Structurally
 * close to McQuestion, but options are keyed on an opaque `key` rather than
 * `word_id` -- an option is frequently an ordinary WordNet term with no
 * `word` row at all (the one-hard-term style), so `word_id ?? 'nota'` (as
 * McQuestion uses) would collide across multiple such options. */
function AnalogyQuestion({ question, result, disabled, onSelect }) {
  const [selected, setSelected] = useState(null)

  function handleClick(key) {
    setSelected(key)
    onSelect(key)
  }

  return (
    <div className="mc-question">
      <p className="mc-prompt">{question.prompt}</p>
      <div className="mc-options">
        {question.analogy_options.map((opt) => (
          <button
            type="button"
            key={opt.key}
            className={optionClass(opt, selected, result)}
            disabled={disabled}
            onClick={() => handleClick(opt.key)}
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

export default AnalogyQuestion
