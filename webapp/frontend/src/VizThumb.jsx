// Small pictograms for the Visualizations gallery cards -- one per chart
// type, drawn in the accent colour, so a card says at a glance what kind of
// picture it opens. Decorative: the card's title carries the meaning.
const W = 120
const H = 64

function Bars() {
  const hs = [18, 44, 30, 52, 24, 38, 14]
  return hs.map((h, i) => <rect key={i} x={10 + i * 15} y={H - 6 - h} width={9} height={h} rx={2} className="vt-fill" />)
}

function Line() {
  return (
    <>
      <polyline points="8,52 24,46 40,48 56,30 72,34 88,14 112,20" className="vt-stroke" />
      <polyline points="8,56 24,54 40,50 56,52 72,44 88,46 112,38" className="vt-stroke vt-soft" />
    </>
  )
}

const NET = [[22, 18], [48, 12], [36, 40], [70, 30], [96, 16], [88, 48], [58, 54]]
function Network() {
  const links = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [3, 5], [5, 6], [2, 6], [4, 5]]
  return (
    <>
      {links.map(([a, b], i) => (
        <line key={i} x1={NET[a][0]} y1={NET[a][1]} x2={NET[b][0]} y2={NET[b][1]} className="vt-stroke vt-thin" />
      ))}
      {NET.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={i === 3 ? 6 : 4} className="vt-fill" />)}
    </>
  )
}

function Scatter() {
  const pts = [[24, 20], [30, 26], [20, 30], [34, 18], [28, 34], [70, 40], [78, 34], [74, 46], [84, 42], [66, 48],
    [96, 16], [102, 22], [92, 24], [52, 20], [48, 48]]
  return pts.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={3} className="vt-fill vt-dot" />)
}

function Tree() {
  const ys = [10, 20, 30, 40, 50, 58]
  return (
    <>
      <polyline points="60,10 44,10 44,20 60,20" className="vt-stroke vt-thin" />
      <polyline points="60,30 50,30 50,40 60,40" className="vt-stroke vt-thin" />
      <polyline points="44,15 30,15 30,35 50,35" className="vt-stroke vt-thin" />
      <polyline points="60,50 38,50 38,58 60,58" className="vt-stroke vt-thin" />
      <polyline points="30,25 16,25 16,54 38,54" className="vt-stroke vt-thin" />
      {ys.map((y) => <rect key={y} x={64} y={y - 2.5} width={44 - (y % 3) * 6} height={5} rx={2} className="vt-fill vt-soft" />)}
    </>
  )
}

function Sunburst() {
  const C = [60, 34]
  const arc = (a0, a1, r0, r1) => {
    const p = (a, r) => `${C[0] + r * Math.cos(a)},${C[1] + r * Math.sin(a)}`
    return `M${p(a0, r1)}A${r1},${r1} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${p(a1, r1)}L${p(a1, r0)}A${r0},${r0} 0 ${a1 - a0 > Math.PI ? 1 : 0} 0 ${p(a0, r0)}Z`
  }
  const seg = [0, 2.3, 3.6, 4.6, 5.4, 6.283]
  return seg.slice(0, -1).map((a, i) => (
    <g key={i}>
      <path d={arc(a - 1.57, seg[i + 1] - 1.57 - 0.04, 10, 19)} className={`vt-fill${i % 2 ? ' vt-soft' : ''}`} />
      <path d={arc(a - 1.57, seg[i + 1] - 1.57 - 0.04, 21, 29)} className="vt-fill vt-soft" />
    </g>
  ))
}

function Field() {
  // the domain map: a cloud, one field picked out
  const pts = []
  for (let i = 0; i < 46; i++) {
    const a = i * 2.39996, r = 4 + Math.sqrt(i) * 3.6
    pts.push([60 + r * Math.cos(a) * 1.5, 32 + r * Math.sin(a)])
  }
  return pts.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={2.2} className={i % 4 === 0 ? 'vt-fill' : 'vt-fill vt-faint'} />)
}

const KINDS = { bars: Bars, line: Line, network: Network, scatter: Scatter, tree: Tree, sunburst: Sunburst, field: Field }

function VizThumb({ kind }) {
  const Shape = KINDS[kind] || Bars
  return (
    <svg className="viz-thumb" viewBox={`0 0 ${W} ${H}`} aria-hidden="true" focusable="false">
      <Shape />
    </svg>
  )
}

export default VizThumb
