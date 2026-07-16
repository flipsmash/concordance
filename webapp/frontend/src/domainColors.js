// Node colors for the word-similarity graph. Keys MUST match
// concordance/usas_domains.py's DOMAIN_BUCKETS keys exactly — the API returns
// `color_bucket` strings that are looked up here directly, no re-mapping.
//
// A validated CVD-safe 6-hue subset (all-pairs check, since any two nodes in a
// force-directed layout can end up adjacent) of the dataviz skill's default
// categorical palette. Plain JS values, not CSS custom properties: canvas fill
// styles need real color strings, and getComputedStyle/CSS vars don't flow
// into ctx.fillStyle for free.
export const DOMAIN_COLORS = {
  mind_language:       { light: '#2a78d6', dark: '#3987e5' },
  people_society:      { light: '#1baf7a', dark: '#199e70' },
  emotion_leisure:     { light: '#eda100', dark: '#c98500' },
  nature_science:      { light: '#008300', dark: '#008300' },
  making_materials:    { light: '#e34948', dark: '#e66767' },
  time_space_commerce: { light: '#eb6834', dark: '#d95926' },
}

// Client-side-only concept for words with no USAS category yet (~90% of the
// corpus today) — not a real bucket, so it's not part of the backend legend.
export const UNCATEGORIZED_GRAY = '#898781'

export function isDark() {
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

export function colorForBucket(bucket) {
  if (!bucket || !DOMAIN_COLORS[bucket]) return UNCATEGORIZED_GRAY
  return DOMAIN_COLORS[bucket][isDark() ? 'dark' : 'light']
}
