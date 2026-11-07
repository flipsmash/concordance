import { useState } from 'react'
import { FAMILY_ORDER, familyColor, routeColor, routeLabel } from './originColors'

const SIZE = 600
const C = SIZE / 2
const R = { inner: [88, 188], outer: [191, 272] }
const SHORT = {
  'Old & Middle English': 'Old/Middle English',
  'Dutch, German & Frankish': 'Dutch & German',
  'Norse & Scandinavian': 'Norse',
  'Arabic, Persian & Turkic': 'Arabic & Persian',
  'Spanish & Portuguese': 'Spanish/Portuguese',
  'Hebrew & Semitic': 'Semitic',
  'Indian languages': 'Indian',
}

function arc(a0, a1, r0, r1) {
  const p = (a, r) => `${(C + r * Math.cos(a)).toFixed(2)},${(C + r * Math.sin(a)).toFixed(2)}`
  const large = a1 - a0 > Math.PI ? 1 : 0
  return `M${p(a0, r1)}A${r1},${r1} 0 ${large} 1 ${p(a1, r1)}L${p(a1, r0)}A${r0},${r0} 0 ${large} 0 ${p(a0, r0)}Z`
}

// Root family (inner ring) > route into English (outer ring), sized by share
// of the words with a known origin. Fixed family order around the ring, so a
// genre filter changes sizes, never positions or colours.
function OriginSunburst({ families, known, selected, onSelect, label }) {
  const [hover, setHover] = useState(null) // {x, y, value, text}
  const byName = Object.fromEntries(families.map((f) => [f.family, f]))
  const ordered = FAMILY_ORDER.filter((f) => byName[f]?.count).map((f) => byName[f])
  const total = known || 1

  const segments = []
  let a = -Math.PI / 2
  for (const f of ordered) {
    const span = (f.count / total) * Math.PI * 2
    segments.push({ kind: 'family', family: f.family, count: f.count, a0: a, a1: a + span })
    let b = a
    for (const r of f.routes) {
      const s = (r.count / total) * Math.PI * 2
      segments.push({ kind: 'route', family: f.family, route: r.route, count: r.count, a0: b, a1: b + s })
      b += s
    }
    a += span
  }

  const isSel = (s) =>
    selected && selected.family === s.family && (s.kind === 'family' ? !selected.route : selected.route === s.route)

  function show(e, s) {
    const box = e.currentTarget.closest('.origin-chart').getBoundingClientRect()
    const pct = ((s.count / total) * 100).toFixed(1)
    setHover({
      x: e.clientX - box.left,
      y: e.clientY - box.top,
      value: `${s.count.toLocaleString()} words · ${pct}%`,
      text: s.kind === 'family' ? s.family : `${s.family} › ${routeLabel(s.route)}`,
    })
  }

  const center = selected
    ? (() => {
        const s = segments.find(isSel)
        return s ? { big: `${((s.count / total) * 100).toFixed(1)}%`, small: s.kind === 'family' ? s.family : routeLabel(s.route) } : null
      })()
    : null

  return (
    <div className="origin-chart">
      {label && <div className="origin-chart-label">{label}</div>}
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="origin-sunburst" role="group" aria-label={`Word origins${label ? `: ${label}` : ''}`}>
        {segments.map((s) => {
          const [r0, r1] = s.kind === 'family' ? R.inner : R.outer
          const fill = s.kind === 'family' ? familyColor(s.family) : routeColor(s.family, s.route)
          const dim = selected && selected.family !== s.family
          return (
            <path
              key={`${s.kind}-${s.family}-${s.route || ''}`}
              d={arc(s.a0, Math.max(s.a0 + 0.0005, s.a1), r0, r1)}
              style={{ fill }}
              className={`origin-seg${isSel(s) ? ' selected' : ''}${dim ? ' dim' : ''}`}
              tabIndex={0}
              role="button"
              aria-label={`${s.kind === 'family' ? s.family : `${s.family}, ${routeLabel(s.route)}`}: ${s.count} words`}
              onMouseMove={(e) => show(e, s)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect(s.kind === 'family' ? { family: s.family } : { family: s.family, route: s.route })}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  onSelect(s.kind === 'family' ? { family: s.family } : { family: s.family, route: s.route })
                }
              }}
            />
          )
        })}
        {segments
          .filter((s) => s.kind === 'family' && s.a1 - s.a0 > 0.2)
          .map((s) => {
            const mid = (s.a0 + s.a1) / 2
            const r = (R.inner[0] + R.inner[1]) / 2
            const name = SHORT[s.family] || s.family
            const room = 2 * r * Math.sin(Math.min(Math.PI / 2, (s.a1 - s.a0) / 2))
            if (name.length * 6.6 > Math.max(room, 70)) return null
            return (
              <text key={`l-${s.family}`} x={C + r * Math.cos(mid)} y={C + r * Math.sin(mid)} className="origin-seg-label">
                {name}
              </text>
            )
          })}
        <text x={C} y={C - 6} className="origin-center-big">
          {center ? center.big : known.toLocaleString()}
        </text>
        <text x={C} y={C + 16} className="origin-center-small">
          {center ? center.small : 'words with a known origin'}
        </text>
      </svg>
      {hover && (
        <div className="origin-tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          <strong>{hover.value}</strong>
          <span>{hover.text}</span>
        </div>
      )}
    </div>
  )
}

export default OriginSunburst
