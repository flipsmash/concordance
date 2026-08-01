// Inline SVG, not <img>, and colored via currentColor rather than the
// hardcoded strokes in images/concordance_logotype*.svg -- this way the
// mark themes itself through whatever `color` the caller sets (typically
// var(--accent-deep)/var(--rail-accent), both of which already flip for
// dark mode) instead of needing separate light/dark asset files kept in
// sync by hand. Geometry reproduced exactly from images/concordance_logotype.svg
// (two circles, r=14, centers 12pt apart, stroke-width 2.5).

export function LogoIcon({ size = 28, className }) {
  return (
    <svg
      width={size}
      height={size * (34 / 46)}
      viewBox="0 0 46 34"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <circle cx="17" cy="17" r="14" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <circle cx="29" cy="17" r="14" fill="none" stroke="currentColor" strokeWidth="2.5" />
    </svg>
  )
}

// viewBox is cropped tight to the actual ink. First attempt measured this
// by rasterizing with ImageMagick + PIL, which silently substitutes a
// different font than "Helvetica, Arial, sans-serif" resolves to in a real
// browser -- the ImageMagick-measured text was ~50px narrower than real
// Chrome/Arial, so that crop clipped the word's tail ("concordan[ce]") in
// production. Re-measured against an actual headless Chrome screenshot
// instead (real x:218-477, y:93-125 out of the source's 680x220 canvas)
// and padded generously beyond that to leave room for other
// browsers/OSes' Arial-fallback metrics differing slightly from Chrome's.
const MARK_ASPECT = 52 / 280

export function LogoMark({ width = 180, className }) {
  return (
    <svg
      width={width}
      height={width * MARK_ASPECT}
      viewBox="210 83 280 52"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="concordance"
    >
      <g transform="translate(240,110)">
        <circle cx="-6" cy="0" r="14" fill="none" stroke="currentColor" strokeWidth="2.5" />
        <circle cx="6" cy="0" r="14" fill="none" stroke="currentColor" strokeWidth="2.5" />
      </g>
      <text
        x="266" y="120"
        fontFamily="Helvetica, Arial, sans-serif"
        fontSize="38" fontWeight="500" letterSpacing="-0.5"
        fill="currentColor"
      >
        concordance
      </text>
    </svg>
  )
}
