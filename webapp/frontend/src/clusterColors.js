import { isDark } from './domainColors'

// Cluster count is data-dependent (currently up to n_clusters=12, see
// concordance/db.py's compute_author_clustering) and could change --
// HSL-rotation-generated rather than hand-picked. domainColors.js's fixed
// 6-hue set is CVD-vetted all-pairs; that doesn't scale to a dozen-plus
// colors by hand, so this is honestly a weaker guarantee, not a from-
// scratch equivalent.
const HUE_STEP = 137 // golden-angle-ish step so adjacent cluster ids don't land on adjacent hues
const SATURATION = 65
const LIGHTNESS = { light: 48, dark: 62 }

export function colorForCluster(clusterId) {
  const hue = (clusterId * HUE_STEP) % 360
  const lightness = LIGHTNESS[isDark() ? 'dark' : 'light']
  return `hsl(${hue}, ${SATURATION}%, ${lightness}%)`
}
