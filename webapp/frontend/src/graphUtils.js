// Shared between GraphView.jsx (2D, eager) and GraphView3D.jsx (3D, lazy) so
// neither the node-sizing math nor the zoomToFit tuning drifts between them.

// zipf ~1 (very rare) -> small node, ~7 ("the") -> large node. Clamped since
// real words rarely hit either extreme.
const ZIPF_MIN = 1
const ZIPF_MAX = 7
const RADIUS_MIN = 4
const RADIUS_MAX = 20

export function radiusForZipf(zipf) {
  const t = Math.min(1, Math.max(0, (zipf - ZIPF_MIN) / (ZIPF_MAX - ZIPF_MIN)))
  return RADIUS_MIN + t * (RADIUS_MAX - RADIUS_MIN)
}

// Canvas fillStyle/Three.js Color can't take a raw CSS var() string — only
// resolved colors. Read the live computed value so it still tracks the app's
// actual light/dark tokens (index.css) instead of duplicating them.
export function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

// zoomToFit(durationMs, paddingPx) — same call, same tuning, in both 2D and 3D.
export const ZOOM_MS = 400
export const ZOOM_PADDING = 40

// A word-graph link's dash by how indirect it is -- the outer ring of its two
// words (0/1 = the word and its direct neighbors). Direct links are solid;
// each ring further out is more broken: long dashes, then shorter, then dots.
// The words themselves stay at full strength whatever the ring.
const LINK_DASHES = [null, null, [6, 2], [4, 3], [2, 3], [1, 3]]

export function linkDash(ring) {
  return LINK_DASHES[Math.min(Math.max(ring, 0), LINK_DASHES.length - 1)]
}

// Word-graph link color: the muted text color at 60% -- dark enough that the
// dash patterns (linkDash) read as different, light enough not to drown the words.
export function linkColor() {
  const c = cssVar('--text', '#6b6375')
  return /^#[0-9a-f]{6}$/i.test(c) ? `${c}99` : c
}
