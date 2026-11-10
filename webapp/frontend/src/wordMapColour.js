// What the "Every word by meaning" map can be coloured by, and the colour of
// each word for a choice. Dataviz rules: a dense scatter can't carry many
// hues, so magnitudes are one hue in five steps (light -> dark; dark mode
// has its own steps so the strongest values stand out against the dark
// surface), part of speech is the three validated categorical hues that pass
// all-pairs (a fourth failed in dark mode), and a many-valued category
// (origin, domain) highlights one value at a time against grey.

// blue ramp, steps 250-650 (light) / 600-200 (dark); see the dataviz palette
const RAMP = {
  light: ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281'],
  dark: ['#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4'],
}
const POS = {
  light: { n: '#2a78d6', a: '#eb6834', v: '#1baf7a' },
  dark: { n: '#3987e5', a: '#d95926', v: '#199e70' },
}
const POS_NAMES = { n: 'noun', a: 'adjective', v: 'verb' }
const HIGHLIGHT = { light: '#2a78d6', dark: '#3987e5' }

export const MEASURES = [
  { id: 'difficulty', label: 'Difficulty', help: 'darker = harder' },
  { id: 'frequency', label: 'How common in print', help: 'darker = more common' },
  { id: 'peak_year', label: 'When it peaked in print', help: 'darker = more recent' },
  { id: 'pos', label: 'Part of speech' },
]

// five bins: difficulty in fixed 20-point bands; the others by quintile, so
// each shade covers a fifth of the words
function bins(values, id) {
  if (id === 'difficulty') return [20, 40, 60, 80]
  const sorted = values.filter((v) => v != null).sort((a, b) => a - b)
  if (!sorted.length) return [0, 0, 0, 0]
  return [0.2, 0.4, 0.6, 0.8].map((q) => sorted[Math.floor(q * (sorted.length - 1))])
}

function stepOf(value, cuts) {
  let i = 0
  while (i < cuts.length && value >= cuts[i]) i++
  return i
}

function fmt(id, v) {
  if (id === 'peak_year') return `${Math.floor(v / 10) * 10}s`
  if (id === 'frequency') return v.toFixed(1)
  return String(Math.round(v))
}

/**
 * colours: one CSS colour per node (null = the map's plain colour),
 * legend: [{ colour, label }] for the key, or null.
 * `choice` is '' (plain), a MEASURES id, 'origin:<i>' or 'domain:<i>'.
 */
export function colourWords(data, choice, dark, grey) {
  const c = data?.colour
  if (!c || !choice) return { colours: null, legend: null }
  const mode = dark ? 'dark' : 'light'
  const n = c.difficulty.length

  if (choice === 'pos') {
    const colours = c.pos.map((p) => POS[mode][p] ?? grey)
    const legend = Object.entries(POS_NAMES).map(([k, name]) => ({ colour: POS[mode][k], label: name }))
    legend.push({ colour: grey, label: 'adverb or other' })
    return { colours, legend }
  }
  if (choice.startsWith('origin:') || choice.startsWith('domain:')) {
    const [field, idx] = choice.split(':')
    const values = c[field]
    const target = Number(idx)
    const colours = values.map((v) => (v === target ? HIGHLIGHT[mode] : grey))
    const count = values.filter((v) => v === target).length
    const name = (field === 'origin' ? c.origins : c.domains)[target]
    return { colours, legend: [{ colour: HIGHLIGHT[mode], label: `${name} (${count.toLocaleString()})` },
      { colour: grey, label: 'everything else' }] }
  }
  const values = c[choice]
  const cuts = bins(values, choice)
  const colours = new Array(n)
  for (let i = 0; i < n; i++) colours[i] = values[i] == null ? grey : RAMP[mode][stepOf(values[i], cuts)]
  const edges = [null, ...cuts, null]
  // print frequency is a log score (zipf) -- meaningless as numbers to a
  // reader -- and its bins are fifths of the vocabulary, so name them so
  const FIFTHS = ['rarest fifth', 'rare', 'middle fifth', 'common', 'most common fifth']
  const legend = RAMP[mode].map((colour, i) => ({
    colour,
    label: choice === 'frequency' ? FIFTHS[i]
      : edges[i] == null ? `under ${fmt(choice, edges[i + 1])}`
        : edges[i + 1] == null ? `${fmt(choice, edges[i])} and up`
          : `${fmt(choice, edges[i])}–${fmt(choice, edges[i + 1])}`,
  }))
  legend.push({ colour: grey, label: 'unknown' })
  return { colours, legend }
}
