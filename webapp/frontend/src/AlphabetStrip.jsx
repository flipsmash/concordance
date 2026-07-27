import './AlphabetStrip.css'

const LETTERS = 'abcdefghijklmnopqrstuvwxyz'.split('')

// Shared by Browse (words), Books, and Authors -- click a letter to jump to
// it, click it again to clear. Callers own what "letter" filters against
// (lemma, title, or author name); this component only renders the strip.
function AlphabetStrip({ letter, onChange }) {
  return (
    <div className="az-strip">
      {LETTERS.map((l) => (
        <button
          type="button"
          key={l}
          className={letter === l ? 'az-letter active' : 'az-letter'}
          onClick={() => onChange(letter === l ? null : l)}
        >
          {l}
        </button>
      ))}
    </div>
  )
}

export default AlphabetStrip
