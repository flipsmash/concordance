import { forwardRef, useMemo } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import SpriteText from 'three-spritetext'
import { colorForBucket } from './domainColors'
import { cssVar, radiusForZipf } from './graphUtils'

// Lazy-loaded from GraphView.jsx (only when the user actually switches to 3D)
// specifically because `three` is a large dependency (WebGL) that 2D users —
// almost certainly the common case — should never have to download.
//
// nodeThreeObject replaces (not extends — see nodeThreeObjectExtend's default
// of false) the library's default sphere with a Group: a lit sphere sized/
// colored the same way the 2D canvas circle is (radiusForZipf,
// colorForBucket) — Lambert, not Basic, so it actually shades under
// 3d-force-graph's default ambient+directional lights instead of rendering as
// a flat uniform-color disc that reads as 2D — a translucent halo standing in
// for 2D's stroke outline on the center node, and a SpriteText label below it
// — a 3D text mesh, since there's no canvas 2D context here to just fillText
// onto.
function buildNode(node, isCenterId) {
  const group = new THREE.Group()
  const isCenter = node.id === isCenterId
  const radius = isCenter ? Math.max(radiusForZipf(node.zipf), 10) : radiusForZipf(node.zipf)

  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 16, 16),
    new THREE.MeshLambertMaterial({ color: colorForBucket(node.color_bucket) }),
  )
  group.add(sphere)

  // 2D's center-node highlight is a thin stroke outline — a wireframe halo
  // here turned out to be dense enough (at typical zoom) to fully occlude
  // the base sphere's color instead of just outlining it. A larger,
  // translucent sphere behind it reads as a halo/glow without hiding the
  // domain color it's supposed to be highlighting.
  if (isCenter) {
    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(radius * 1.4, 16, 16),
      new THREE.MeshBasicMaterial({
        color: cssVar('--text-h', '#08060d'),
        transparent: true,
        opacity: 0.25,
        depthWrite: false,
      }),
    )
    group.add(halo)
  }

  const label = new SpriteText(node.lemma)
  label.textHeight = 4
  label.color = cssVar('--text-h', '#08060d')
  label.position.set(0, -(radius + 5), 0)
  group.add(label)

  return group
}

const GraphView3D = forwardRef(function GraphView3D(
  { graphData, center, width, height, nodeLabel, onNodeClick, onEngineStop },
  ref,
) {
  // Re-run only when the center node changes — not on every render — since
  // building a fresh Group per node on each frame would fight the engine.
  const nodeThreeObject = useMemo(() => {
    const centerId = center?.id
    return (node) => buildNode(node, centerId)
  }, [center?.id])

  return (
    <ForceGraph3D
      ref={ref}
      width={width}
      height={height}
      graphData={graphData}
      backgroundColor={cssVar('--bg', '#ffffff')}
      // orbit constrains drag-to-rotate to azimuth/polar around the graph's
      // center — the explicit rotate buttons in GraphView.jsx drive the same
      // two axes via camera()/cameraPosition(), so drag and buttons behave
      // consistently. trackball (the default) tumbles freely and doesn't
      // match "rotate along x and y" either physically or in how the buttons
      // work.
      controlType="orbit"
      nodeLabel={nodeLabel}
      nodeThreeObject={nodeThreeObject}
      linkColor={() => cssVar('--border', '#e5e4e7')}
      onNodeClick={onNodeClick}
      onEngineStop={onEngineStop}
    />
  )
})

export default GraphView3D
