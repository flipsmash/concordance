import { isDark } from './domainColors'

// Sequential ramp for the similarity matrices: one hue (the validated default
// blue scale), more overlap = further from the surface -- darker on light,
// brighter on dark, with dark mode given its own steps rather than reusing
// the light ones over a dark background (which drew the old matrix nearly
// black). Cells used to be the accent colour at alpha = raw score, and with
// most scores between 0.05 and 0.3 the whole matrix read as one pale wash.
const RAMP = {
  light: ['#eef4fc', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'],
  dark: ['#1a2230', '#163252', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'],
}
const SELF = { light: '#d9dcdf', dark: '#3a4049' }

export function matrixRamp() {
  return RAMP[isDark() ? 'dark' : 'light']
}

export function selfColor() {
  return SELF[isDark() ? 'dark' : 'light']
}

// The score range the ramp spans: the 2nd..99th percentile of the off-
// diagonal scores, so a few near-duplicate pairs don't flatten everything
// else into the bottom step.
export function scoreDomain(grid) {
  const vals = []
  for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid.length; j++) {
      if (i !== j) vals.push(grid[i][j].score)
    }
  }
  if (!vals.length) return [0, 1]
  vals.sort((a, b) => a - b)
  const lo = vals[Math.floor(vals.length * 0.02)]
  const hi = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.99))]
  return hi > lo ? [lo, hi] : [0, Math.max(hi, 1e-6)]
}

export function cellColor(score, [lo, hi], ramp) {
  const t = Math.max(0, Math.min(1, (score - lo) / (hi - lo)))
  return ramp[Math.min(ramp.length - 1, Math.floor(t * ramp.length))]
}
