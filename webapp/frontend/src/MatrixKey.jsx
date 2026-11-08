import { matrixRamp } from './matrixColors'
import './MatrixKey.css'

// The matrices' colour key: the ramp's steps, least to most shared rare
// vocabulary (scaled to this matrix's own range -- see scoreDomain).
function MatrixKey() {
  const ramp = matrixRamp()
  return (
    <div className="matrix-key" aria-label="Colour key: lighter means less shared vocabulary, darker means more">
      <span className="muted">Less shared vocabulary</span>
      <span className="matrix-key-steps">
        {ramp.map((c) => <span key={c} style={{ background: c }} />)}
      </span>
      <span className="muted">More</span>
    </div>
  )
}

export default MatrixKey
