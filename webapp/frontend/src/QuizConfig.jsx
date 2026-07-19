import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import MultiSelect from './MultiSelect'
import './QuizConfig.css'

const API_BASE = ''
const LENGTH_PRESETS = [5, 10, 20]
const QUESTION_TYPES = [
  { value: 'mc', label: 'Multiple choice' },
  { value: 'true_false', label: 'True / False' },
  { value: 'matching', label: 'Matching' },
]

function QuizConfig() {
  const navigate = useNavigate()
  const [meta, setMeta] = useState(null)
  const [metaError, setMetaError] = useState('')

  const [length, setLength] = useState(10)
  const [customLength, setCustomLength] = useState('')
  const [useCustomLength, setUseCustomLength] = useState(false)
  const [types, setTypes] = useState(['mc'])
  const [mcChoiceCount, setMcChoiceCount] = useState(4)
  const [matchingSetSize, setMatchingSetSize] = useState(4)
  const [direction, setDirection] = useState('definition_to_word')
  const [notaEnabled, setNotaEnabled] = useState(false)
  const [notaRate, setNotaRate] = useState(15)
  const [difficultyMin, setDifficultyMin] = useState('')
  const [difficultyMax, setDifficultyMax] = useState('')
  const [pos, setPos] = useState([])
  const [domains, setDomains] = useState([])
  const [smartRatio, setSmartRatio] = useState(70)
  const [weights, setWeights] = useState({ orthographic: 34, semantic: 33, domain: 33 })
  const [srEnabled, setSrEnabled] = useState(false)
  const [srFrequency, setSrFrequency] = useState('normal')

  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/api/quiz/meta`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error('failed to load options'))))
      .then(setMeta)
      .catch((err) => setMetaError(err.message))
  }, [])

  function handleWeightChange(key, value) {
    setWeights((w) => ({ ...w, [key]: value }))
  }

  function toggleType(value) {
    setTypes((cur) => (cur.includes(value) ? cur.filter((t) => t !== value) : [...cur, value]))
  }

  function handleSubmit(e) {
    e.preventDefault()
    const effectiveLength = useCustomLength ? parseInt(customLength, 10) : length
    if (!effectiveLength || effectiveLength < 1) {
      setError('enter a valid question count')
      return
    }
    if (types.length === 0) {
      setError('pick at least one question type')
      return
    }
    const weightTotal = weights.orthographic + weights.semantic + weights.domain || 1
    setStarting(true)
    setError('')
    fetch(`${API_BASE}/api/quiz/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        length: effectiveLength,
        types,
        mc_choice_count: mcChoiceCount,
        matching_set_size: matchingSetSize,
        direction,
        nota_enabled: notaEnabled,
        nota_rate: notaRate / 100,
        difficulty_min: difficultyMin === '' ? null : Number(difficultyMin),
        difficulty_max: difficultyMax === '' ? null : Number(difficultyMax),
        pos: pos.length ? pos : null,
        domains: domains.length ? domains : null,
        spaced_repetition_enabled: srEnabled,
        spaced_repetition_frequency: srFrequency,
        smart_vs_random_ratio: smartRatio / 100,
        strategy_weights: {
          orthographic: weights.orthographic / weightTotal,
          semantic: weights.semantic / weightTotal,
          domain: weights.domain / weightTotal,
          antonym: 0,
        },
      }),
    })
      .then((res) => (res.ok ? res.json() : res.json().then((d) => Promise.reject(new Error(d.detail || 'could not start quiz')))))
      .then((data) => navigate(`/app/quiz/${data.session_id}`))
      .catch((err) => setError(err.message))
      .finally(() => setStarting(false))
  }

  return (
    <div className="quiz-config-page">
      <h1>Take a quiz</h1>
      {metaError && <div className="error-banner">{metaError}</div>}

      <form className="quiz-config-form" onSubmit={handleSubmit}>
        <fieldset>
          <legend>Length</legend>
          <div className="quiz-pill-row">
            {LENGTH_PRESETS.map((n) => (
              <button
                type="button"
                key={n}
                className={!useCustomLength && length === n ? 'quiz-pill active' : 'quiz-pill'}
                onClick={() => {
                  setLength(n)
                  setUseCustomLength(false)
                }}
              >
                {n}
              </button>
            ))}
            <button
              type="button"
              className={useCustomLength ? 'quiz-pill active' : 'quiz-pill'}
              onClick={() => setUseCustomLength(true)}
            >
              Custom
            </button>
            {useCustomLength && (
              <input
                type="number"
                min="1"
                max="100"
                className="quiz-pill-input"
                value={customLength}
                onChange={(e) => setCustomLength(e.target.value)}
                placeholder="#"
                autoFocus
              />
            )}
          </div>
        </fieldset>

        <fieldset>
          <legend>Question types</legend>
          <div className="quiz-radio-row">
            {QUESTION_TYPES.map((t) => (
              <label className="quiz-radio" key={t.value}>
                <input type="checkbox" checked={types.includes(t.value)} onChange={() => toggleType(t.value)} />
                {t.label}
              </label>
            ))}
          </div>
          <p className="quiz-config-hint">
            Pick one for a single-type test, or several to blend them within one test.
          </p>
        </fieldset>

        <fieldset>
          <legend>Direction</legend>
          <div className="quiz-radio-row">
            <label className="quiz-radio">
              <input
                type="radio"
                name="direction"
                checked={direction === 'definition_to_word'}
                onChange={() => setDirection('definition_to_word')}
              />
              Show the definition, pick the word
            </label>
            <label className="quiz-radio">
              <input
                type="radio"
                name="direction"
                checked={direction === 'word_to_definition'}
                onChange={() => setDirection('word_to_definition')}
              />
              Show the word, pick the definition
            </label>
          </div>
          <p className="quiz-config-hint">Applies to multiple choice and true/false; matching always shows both.</p>
        </fieldset>

        <fieldset>
          <legend>Filters</legend>
          <div className="quiz-filter-row">
            <MultiSelect label="Part of speech" options={meta?.pos_values ?? []} selected={pos} onChange={setPos} />
            <MultiSelect
              label="Domain"
              options={meta?.domains?.map((d) => d.name) ?? []}
              selected={domains
                .map((bucket) => meta?.domains?.find((d) => d.bucket === bucket)?.name)
                .filter(Boolean)}
              onChange={(names) =>
                setDomains(names.map((name) => meta?.domains?.find((d) => d.name === name)?.bucket).filter(Boolean))
              }
            />
          </div>
          <div className="quiz-difficulty-row">
            <label>
              Difficulty min
              <input
                type="number"
                min="0"
                max="100"
                value={difficultyMin}
                onChange={(e) => setDifficultyMin(e.target.value)}
                placeholder="0"
              />
            </label>
            <label>
              Difficulty max
              <input
                type="number"
                min="0"
                max="100"
                value={difficultyMax}
                onChange={(e) => setDifficultyMax(e.target.value)}
                placeholder="100"
              />
            </label>
          </div>
        </fieldset>

        <fieldset>
          <legend>Choices</legend>
          <div className="quiz-filter-row">
            {types.includes('mc') && (
              <label className="quiz-inline-number">
                Options per question
                <input
                  type="number"
                  min="2"
                  max="8"
                  value={mcChoiceCount}
                  onChange={(e) => setMcChoiceCount(Number(e.target.value))}
                />
              </label>
            )}
            {types.includes('matching') && (
              <label className="quiz-inline-number">
                Pairs per matching set
                <input
                  type="number"
                  min="2"
                  max="8"
                  value={matchingSetSize}
                  onChange={(e) => setMatchingSetSize(Number(e.target.value))}
                />
              </label>
            )}
          </div>
          {types.includes('mc') && (
            <>
              <label className="quiz-checkbox">
                <input type="checkbox" checked={notaEnabled} onChange={(e) => setNotaEnabled(e.target.checked)} />
                Include "None of the above" (multiple choice only)
              </label>
              {notaEnabled && (
                <label className="quiz-inline-number">
                  Correct rate
                  <input
                    type="number"
                    min="0"
                    max="100"
                    value={notaRate}
                    onChange={(e) => setNotaRate(Number(e.target.value))}
                  />
                  %
                </label>
              )}
            </>
          )}
        </fieldset>

        <fieldset>
          <legend>Spaced repetition</legend>
          <label className="quiz-checkbox">
            <input type="checkbox" checked={srEnabled} onChange={(e) => setSrEnabled(e.target.checked)} />
            Prioritize words I've missed before
          </label>
          {srEnabled && (
            <div className="quiz-pill-row">
              {['loose', 'normal', 'tight'].map((f) => (
                <button
                  type="button"
                  key={f}
                  className={srFrequency === f ? 'quiz-pill active' : 'quiz-pill'}
                  onClick={() => setSrFrequency(f)}
                >
                  {f}
                </button>
              ))}
            </div>
          )}
          <p className="quiz-config-hint">
            Missed words resurface sooner than ones you got right — always tracked in the background;
            this just controls whether new quizzes lean on that history and how aggressively.
          </p>
        </fieldset>

        <details className="quiz-advanced">
          <summary>Advanced: distractor mix</summary>
          <label className="quiz-slider-row">
            <span>Smart vs. random distractors</span>
            <input
              type="range"
              min="0"
              max="100"
              value={smartRatio}
              onChange={(e) => setSmartRatio(Number(e.target.value))}
            />
            <span className="quiz-slider-value">{smartRatio}% smart</span>
          </label>
          {[
            ['orthographic', 'Looks/spells alike'],
            ['semantic', 'Close in meaning'],
            ['domain', 'Same subject area'],
          ].map(([key, label]) => (
            <label className="quiz-slider-row" key={key}>
              <span>{label}</span>
              <input
                type="range"
                min="0"
                max="100"
                value={weights[key]}
                onChange={(e) => handleWeightChange(key, Number(e.target.value))}
              />
              <span className="quiz-slider-value">{weights[key]}</span>
            </label>
          ))}
        </details>

        {error && <div className="error-banner">{error}</div>}

        <button type="submit" className="quiz-start-btn" disabled={starting}>
          {starting ? 'Starting…' : 'Start quiz'}
        </button>
      </form>
    </div>
  )
}

export default QuizConfig
