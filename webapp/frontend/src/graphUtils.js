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
