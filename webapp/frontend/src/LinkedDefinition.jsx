import { Link } from 'react-router-dom'

// Renders `text` (a word's own definition) with any substring matching one
// of `links` (word_detail's definition_links -- other active vocab words
// this definition mentions, computed by `concordance link-definitions`)
// turned into a clickable Link to that word's own detail page. `surface` is
// the literal inflected form recorded at link-computation time ("proscribed"),
// not necessarily the target's own headword spelling ("proscribe") -- match
// on that exact substring, case-insensitively, whole-word only.
//
// Escaping surfaces before building the regex is defensive but always a
// no-op in practice: a surface only ever comes from a spaCy tok.is_alpha
// token (see db.compute_definition_links), so it can never itself contain a
// regex metacharacter.
function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function LinkedDefinition({ text, links }) {
  if (!links || links.length === 0) return text

  const pattern = links.map((l) => escapeRegExp(l.surface)).join('|')
  const re = new RegExp(`\\b(${pattern})\\b`, 'gi')
  const bySurfaceLower = new Map(links.map((l) => [l.surface.toLowerCase(), l]))

  const pieces = text.split(re)
  return pieces.map((piece, i) => {
    const link = bySurfaceLower.get(piece.toLowerCase())
    if (!link) return piece
    return (
      <Link key={`${i}-${link.word_id}`} to={`/app/words/${link.word_id}`} className="definition-link">
        {piece}
      </Link>
    )
  })
}

export default LinkedDefinition
