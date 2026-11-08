import { Link } from 'react-router-dom'
import { NON_LANGUAGE, familyColor } from './originColors'

// The word page's "Origin" section: origins.py's answer (word_origin) in a
// sentence, with the family's swatch from the Origins chart so the two read
// as one picture, and where the answer came from. A word no dictionary
// source covers says so -- origins are never guessed.

function originSentence({ root_lang_name: lang, family, route }) {
  if (family === 'Mixed') return 'Formed in English from parts of different languages.'
  if (family === 'Obscure origin') {
    return route === 'formed' ? 'Formed in English from a word of obscure origin.' : 'Of obscure origin.'
  }
  if (family === 'Imitative') {
    return route === 'formed' ? 'Formed in English from an imitative word.' : 'Imitative: it echoes a sound.'
  }
  if (route === 'inherited') return `Inherited from ${lang}.`
  if (route === 'direct') return `Borrowed from ${lang}.`
  if (route === 'formed') return `Formed in English from ${lang} roots.`
  if (route.startsWith('via:')) return `From ${lang}, by way of ${route.slice(4)}.`
  return `From ${lang}.`
}

const METHOD_NOTE = {
  wiktionary: 'Wiktionary’s etymology',
  'wiktionary-text': 'Wiktionary’s etymology',
  '0dict': 'the 0 Dict’s etymology',
  '0dict-formation': 'the 0 Dict, which forms it from a neighbouring entry',
  pointer: 'the spelling its definition points to',
  derivative: 'the word it’s built on',
  compound: 'the words it’s made of',
}

function WordOrigin({ origin }) {
  return (
    <section className="word-detail-section">
      <h2>Origin</h2>
      {origin ? (
        <>
          <p className="word-origin-line">
            <span className="word-origin-swatch" style={{ background: familyColor(origin.family) }} aria-hidden="true" />
            {originSentence(origin)}
            {!NON_LANGUAGE.includes(origin.family) && origin.family !== origin.root_lang_name && (
              <span className="muted"> ({origin.family})</span>
            )}
          </p>
          <p className="muted word-origin-source">
            Source: {METHOD_NOTE[origin.method] || origin.method}.{' '}
            <Link to="/app/visualizations/origins">Where all the words come from →</Link>
          </p>
        </>
      ) : (
        <p className="muted">No dictionary source records where this word comes from.</p>
      )}
    </section>
  )
}

export default WordOrigin
