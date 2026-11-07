import { useState } from 'react'
import { familyColor } from './originColors'

const W = 800
const H = 520
const GAP = 2

// Squarified treemap layout (Bruls, Huizing & van Wijk): rows of items laid
// along the shorter side, each row kept as square-ish as possible.
function squarify(items, x, y, w, h) {
  const out = []
  const total = items.reduce((s, it) => s + it.value, 0)
  if (!total || w <= 0 || h <= 0) return out
  const scale = (w * h) / total
  let rest = items.map((it) => ({ ...it, area: it.value * scale }))
  let rx = x, ry = y, rw = w, rh = h
  while (rest.length) {
    const side = Math.min(rw, rh)
    let row = [rest[0]]
    let i = 1
    const worst = (r) => {
      const s = r.reduce((a, b) => a + b.area, 0)
      const mx = Math.max(...r.map((q) => q.area)), mn = Math.min(...r.map((q) => q.area))
      return Math.max((side * side * mx) / (s * s), (s * s) / (side * side * mn))
    }
    while (i < rest.length && worst([...row, rest[i]]) <= worst(row)) row.push(rest[i++])
    const s = row.reduce((a, b) => a + b.area, 0)
    const thick = s / side
    let off = 0
    for (const r of row) {
      const len = r.area / thick
      out.push(rw >= rh
        ? { ...r, x: rx, y: ry + off, w: thick, h: len }
        : { ...r, x: rx + off, y: ry, w: len, h: thick })
      off += len
    }
    if (rw >= rh) { rx += thick; rw -= thick } else { ry += thick; rh -= thick }
    rest = rest.slice(i)
  }
  return out
}

function fits(text, w, h, px = 6.4) {
  return w - 8 >= text.length * px && h >= 18
}

// Root family > root language, sized by share of words with a known origin.
function OriginTreemap({ families, known, selected, onSelect, label }) {
  const [hover, setHover] = useState(null)
  const total = known || 1
  const fams = squarify(
    families.filter((f) => f.count).map((f) => ({ key: f.family, value: f.count, family: f })),
    0, 0, W, H,
  )

  function show(e, value, text) {
    const box = e.currentTarget.closest('.origin-chart').getBoundingClientRect()
    setHover({ x: e.clientX - box.left, y: e.clientY - box.top, value, text })
  }
  const pct = (n) => `${((n / total) * 100).toFixed(1)}%`

  return (
    <div className="origin-chart">
      {label && <div className="origin-chart-label">{label}</div>}
      <svg viewBox={`0 0 ${W} ${H}`} className="origin-treemap" role="group" aria-label={`Word origins treemap${label ? `: ${label}` : ''}`}>
        {fams.map((fr) => {
          const f = fr.family
          const color = familyColor(f.family)
          const headText = `${f.family} · ${pct(f.count)}`
          const head = fits(headText, fr.w, fr.h, 7) && fr.h > 40 ? 18 : 0
          const langs = squarify(
            f.languages.map((l) => ({ key: l.code, value: l.count, lang: l })),
            fr.x + GAP / 2, fr.y + GAP / 2 + head, fr.w - GAP, fr.h - GAP - head,
          )
          const dim = selected && selected.family !== f.family
          return (
            <g key={f.family} className={dim ? 'dim' : ''}>
              <rect x={fr.x + GAP / 2} y={fr.y + GAP / 2} width={Math.max(0, fr.w - GAP)} height={Math.max(0, fr.h - GAP)}
                style={{ fill: color }} className="origin-tm-family" />
              {head > 0 && (
                <text x={fr.x + 6} y={fr.y + 14} className="origin-tm-family-label">
                  {headText}
                </text>
              )}
              {langs.map((lr, i) => {
                const l = lr.lang
                const sel = selected && selected.family === f.family && selected.lang === l.code
                return (
                  <g key={l.code}>
                    <rect
                      x={lr.x + GAP / 2} y={lr.y + GAP / 2}
                      width={Math.max(0, lr.w - GAP)} height={Math.max(0, lr.h - GAP)}
                      style={{ fill: i % 2 ? `color-mix(in oklch, ${color}, var(--bg) 22%)` : color }}
                      className={`origin-seg${sel ? ' selected' : ''}`}
                      tabIndex={0}
                      role="button"
                      aria-label={`${l.name} (${f.family}): ${l.count} words`}
                      onMouseMove={(e) => show(e, `${l.count.toLocaleString()} words · ${pct(l.count)}`, `${l.name} — ${f.family}`)}
                      onMouseLeave={() => setHover(null)}
                      onClick={() => onSelect({ family: f.family, lang: l.code, langName: l.name })}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          onSelect({ family: f.family, lang: l.code, langName: l.name })
                        }
                      }}
                    />
                    {fits(l.name, lr.w, lr.h, 6) && (
                      <text x={lr.x + 6} y={lr.y + 14} className="origin-tm-lang-label">{l.name}</text>
                    )}
                  </g>
                )
              })}
            </g>
          )
        })}
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

export default OriginTreemap
