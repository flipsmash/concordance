import { useCallback, useState } from 'react'

/** Shared domain/difficulty/exclusivity click-to-filter state for the book-
 * and author-detail pages. Owns three independent axes -- selectedDomain (a
 * bucket key, 'uncategorized', or null), selectedBand (a {min, max} string
 * pair, 'unscored', or null), and exclusiveOnly (boolean) -- and projects
 * them into the three different param shapes /api/browse/words,
 * /api/browse/domain-summary, and /api/browse/difficulty-bands each need.
 * The three shapes deliberately diverge: domain-summary never receives its
 * own axis (domain/uncategorized) and difficulty-bands never receives its
 * own axis (difficulty_min/max/unscored_only), since each endpoint computes
 * that axis as ITS OWN output -- passing it back in as a filter too would
 * either 400 (browse_words does reject the combination) or just be
 * meaningless. `exclusive` has no such restriction (it doesn't compute a
 * per-word bucket/band the way domain/difficulty do), so it passes through
 * to all three unchanged. Getting this projection wrong silently breaks the
 * charts' cross-filtering (each chart's counts should reflect the OTHER
 * axes' active selection) without any visible error, which is why it's
 * centralized here instead of re-derived per page.
 *
 * Band bounds are kept as STRINGS, not numbers: usePagedTable's extraParams
 * loop does `else if (value) params.set(...)`, which drops a numeric 0 as
 * falsy -- a real band_min for the lowest difficulty band. */
export function useWordFilters() {
  const [selectedDomain, setSelectedDomain] = useState(null)
  const [selectedBand, setSelectedBand] = useState(null)
  const [exclusiveOnly, setExclusiveOnly] = useState(false)

  const toggleDomain = useCallback((bucket) => {
    setSelectedDomain((d) => (d === bucket ? null : bucket))
  }, [])

  const toggleBand = useCallback((band) => {
    if (band.band_min === null) {
      setSelectedBand((b) => (b === 'unscored' ? null : 'unscored'))
      return
    }
    const min = String(band.band_min)
    const max = String(band.band_max)
    setSelectedBand((b) => (b && b !== 'unscored' && b.min === min && b.max === max ? null : { min, max }))
  }, [])

  const toggleExclusive = useCallback(() => {
    setExclusiveOnly((e) => !e)
  }, [])

  const clear = useCallback(() => {
    setSelectedDomain(null)
    setSelectedBand(null)
    setExclusiveOnly(false)
  }, [])

  const isUncategorized = selectedDomain === 'uncategorized'
  const isUnscored = selectedBand === 'unscored'
  const band = selectedBand && !isUnscored ? selectedBand : null

  const wordParams = {
    domain: selectedDomain && !isUncategorized ? [selectedDomain] : [],
    uncategorized: isUncategorized,
    difficulty_min: band ? band.min : '',
    difficulty_max: band ? band.max : '',
    unscored_only: isUnscored,
    exclusive: exclusiveOnly,
  }

  const domainSummaryParams = {
    difficulty_min: wordParams.difficulty_min,
    difficulty_max: wordParams.difficulty_max,
    unscored_only: wordParams.unscored_only,
    exclusive: exclusiveOnly,
  }

  const bandsParams = {
    domain: wordParams.domain,
    uncategorized: wordParams.uncategorized,
    exclusive: exclusiveOnly,
  }

  return {
    selectedDomain,
    selectedBand,
    exclusiveOnly,
    toggleDomain,
    toggleBand,
    toggleExclusive,
    clear,
    wordParams,
    domainSummaryParams,
    bandsParams,
  }
}
