import { forwardRef, useMemo } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import SpriteText from 'three-spritetext'
import { colorForBucket } from './domainColors'
import { cssVar, linkDash, radiusForZipf } from './graphUtils'

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
    new THREE.MeshLambertMaterial({
      color: isCenter ? cssVar('--graph-center', '#e6d200') : colorForBucket(node.color_bucket),
    }),
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
  label.material.transparent = true
  label.position.set(0, -(radius + 5), 0)
  group.add(label)

  return group
}

const GraphView3D = forwardRef(function GraphView3D(
  { graphData, center, width, height, nodeLabel, onNodeClick, onEngineStop, linkRing },
  ref,
) {
  // Re-run only when the center node changes — not on every render — since
  // building a fresh Group per node on each frame would fight the engine.
  const nodeThreeObject = useMemo(() => {
    const centerId = center?.id
    return (node) => buildNode(node, centerId)
  }, [center?.id])

  // Same dashes as the 2D canvas (graphUtils.linkDash): 3d-force-graph has no
  // dashed links, so each link is its own THREE.Line with a dashed material,
  // its two ends moved (and dash distances recomputed) as the layout runs.
  const linkThreeObject = useMemo(() => (link) => {
    const color = cssVar('--text', '#6b6375')   // the 2D links' color (graphUtils.linkColor); opacity set below
    const dash = linkDash(linkRing ? linkRing(link) : 1)
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3))
    const material = dash
      ? new THREE.LineDashedMaterial({ color, dashSize: dash[0], gapSize: dash[1], transparent: true, opacity: 0.6 })
      : new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 })
    return new THREE.Line(geometry, material)
  }, [linkRing])

  const linkPositionUpdate = (line, { start, end }) => {
    const pos = line.geometry.attributes.position
    pos.setXYZ(0, start.x, start.y, start.z)
    pos.setXYZ(1, end.x, end.y, end.z)
    pos.needsUpdate = true
    line.computeLineDistances()
    return true
  }

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
      linkThreeObject={linkThreeObject}
      linkPositionUpdate={linkPositionUpdate}
      onNodeClick={onNodeClick}
      onEngineStop={onEngineStop}
    />
  )
})

export default GraphView3D
