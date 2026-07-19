import { useMemo, useState } from 'react'

/** Two independently-shuffled columns -- words and definitions -- paired by
 * clicking a word then its matching definition. The server never tells the
 * client which slot pairs with which word (that's the whole answer key), so
 * there's nothing here to derive a shortcut from. */
function MatchingQuestion({ question, result, disabled, onSubmit }) {
  const [pairing, setPairing] = useState({}) // word_id -> definition_slot
  const [activeWordId, setActiveWordId] = useState(null)

  const usedSlots = useMemo(() => new Set(Object.values(pairing)), [pairing])
  const pairResultByWord = useMemo(() => {
    if (!result?.pair_results) return null
    return Object.fromEntries(result.pair_results.map((p) => [p.word_id, p]))
  }, [result])

  function pickWord(wordId) {
    if (disabled) return
    setActiveWordId((cur) => (cur === wordId ? null : wordId))
  }

  function pickSlot(slot) {
    if (disabled || usedSlots.has(slot)) return
    if (activeWordId == null) return
    setPairing((p) => ({ ...p, [activeWordId]: slot }))
    setActiveWordId(null)
  }

  function unpair(wordId) {
    if (disabled) return
    setPairing((p) => {
      const next = { ...p }
      delete next[wordId]
      return next
    })
  }

  function handleSubmit() {
    const pairs = question.word_slots.map((w) => ({ word_id: w.word_id, definition_slot: pairing[w.word_id] }))
    onSubmit(pairs)
  }

  const allPaired = question.word_slots.every((w) => pairing[w.word_id])

  return (
    <div className="matching-question">
      <p className="mc-prompt">Match each word to its definition.</p>
      <div className="matching-columns">
        <div className="matching-column">
          {question.word_slots.map((w) => {
            const paired = pairing[w.word_id]
            const pr = pairResultByWord?.[w.word_id]
            let cls = 'matching-item'
            if (pr) cls += pr.is_correct ? ' reveal-correct' : ' reveal-incorrect'
            else if (activeWordId === w.word_id) cls += ' active'
            else if (paired) cls += ' paired'
            return (
              <button type="button" key={w.word_id} className={cls} disabled={disabled}
                onClick={() => (paired ? unpair(w.word_id) : pickWord(w.word_id))}>
                {w.lemma}
                {paired && <span className="matching-slot-badge">{paired}</span>}
              </button>
            )
          })}
        </div>
        <div className="matching-column">
          {question.definition_slots.map((d) => (
            <button type="button" key={d.slot}
              className={usedSlots.has(d.slot) ? 'matching-item paired' : 'matching-item'}
              disabled={disabled || usedSlots.has(d.slot)} onClick={() => pickSlot(d.slot)}>
              <span className="matching-slot-badge">{d.slot}</span> {d.quiz_definition}
            </button>
          ))}
        </div>
      </div>
      {!result && (
        <button type="button" className="quiz-next-btn" disabled={disabled || !allPaired} onClick={handleSubmit}>
          Submit matches
        </button>
      )}
      {result && (
        <p className={result.pair_results.every((p) => p.is_correct) ? 'mc-feedback correct' : 'mc-feedback incorrect'}>
          {result.pair_results.filter((p) => p.is_correct).length} of {result.pair_results.length} correct.
        </p>
      )}
    </div>
  )
}

export default MatchingQuestion
