// Tooltip text for the archaic-currency tag (Browse.jsx, AuthorWorks.jsx,
// WorkDetail.jsx) -- one shared source so the three copies of the tag can't
// drift. Ordinal per concordance/archaic.py: current < dated < archaic <
// obsolete; each description is phrased relative to its neighbors rather
// than standalone, since that ordering is the point.
export const ARCHAIC_TOOLTIPS = {
  dated: 'Still in use, but sounds old-fashioned today -- milder than archaic or obsolete.',
  archaic: 'Distinctly old-fashioned, evoking an earlier era -- stronger than dated, short of obsolete.',
  obsolete: 'No longer in use at all -- the strongest of the three, found only in older texts.',
}
