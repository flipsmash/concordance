const LABELS = {
  accepted: 'In concordance',
  pruned: 'Pruned from concordance',
  rejected: 'Rejected in concordance',
  unique: 'Not in concordance',
}

// Renders nothing for null/unchecked -- an entry that hasn't been through
// `oed-concordance-match` yet (or isn't a lemma entry, so was never in
// scope for it) shouldn't look like a settled verdict.
function OedMatchBadge({ match }) {
  if (!match || !LABELS[match]) return null
  return <span className={`oed-match-badge match-${match}`}>{LABELS[match]}</span>
}

export default OedMatchBadge
