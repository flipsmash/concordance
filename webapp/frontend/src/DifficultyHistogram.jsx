import './DifficultyHistogram.css'

const BAR_AREA_HEIGHT = 120 // px

// An actual histogram: bar HEIGHT encodes frequency (word_count), on a
// shared baseline, with equal-width bins along the x-axis -- not the main
// Browse page's proportional-WIDTH filter strip (a different, deliberately
// compact widget for a different job: click-to-set-a-range inline in a
// facet row). That one stays as-is; this is its own component because
// reusing it here rendered every band the same height, which isn't a
// histogram at all, just a segmented bar.
//
// Every column, including "Not yet scored", is clickable -- unlike Browse's
// own histogram (which guards that column out since it has no way to
// express "unscored" as a difficulty_min/max range), this one's parent can
// pass the backend's `unscored_only` filter instead.
// unit: what a bar counts ("words" by default; the hub's charts count books
// or authors). hint: a line under the bars saying what a click does.
// headline: (band, percent) => text -- when given, a bar holding over half
// the total and at least 4x the next one is pulled out as a sentence above
// the chart (still clickable) and the rest are scaled on their own, so one
// giant bar (21,493 books with 0 unique words) no longer flattens the others.
function DifficultyHistogram({ bands, selectedBand, onSelect, unit = 'words', hint = null, headline = null }) {
  // null = still loading: hold the chart's height and say so, rather than
  // render nothing (a slow request used to look like a missing chart)
  if (bands === null) return <div className="diff-hist diff-hist-loading muted">Loading…</div>
  if (bands.length === 0) return null
  const total = bands.reduce((s, b) => s + b.word_count, 0)
  const sorted = [...bands].sort((a, b) => b.word_count - a.word_count)
  const dominant =
    headline && sorted.length > 1 && sorted[0].word_count > total / 2 && sorted[0].word_count >= 4 * sorted[1].word_count
      ? sorted[0]
      : null
  const shown = dominant ? bands.filter((b) => b !== dominant) : bands
  const maxCount = Math.max(...shown.map((b) => b.word_count), 1)

  function isSelected(b) {
    if (b.band_min === null) return selectedBand === 'unscored'
    return (
      selectedBand &&
      selectedBand !== 'unscored' &&
      selectedBand.min === String(b.band_min) &&
      selectedBand.max === String(b.band_max)
    )
  }

  return (
    <div className="diff-hist">
      {dominant && (
        <button type="button" className="diff-hist-headline" onClick={() => onSelect(dominant)}>
          {headline(dominant, Math.round((dominant.word_count / total) * 100))}
        </button>
      )}
      <div className="diff-hist-bars">
        {shown.map((b) => {
          const heightPx = b.word_count === 0 ? 0 : Math.max((b.word_count / maxCount) * BAR_AREA_HEIGHT, 3)
          return (
            <button
              type="button"
              className={isSelected(b) ? 'diff-hist-col active' : 'diff-hist-col'}
              key={b.label}
              title={`${b.label}: ${b.word_count.toLocaleString()} ${unit}`}
              onClick={() => onSelect(b)}
            >
              <span className="diff-hist-count">{b.word_count > 0 ? b.word_count.toLocaleString() : ''}</span>
              <div className="diff-hist-bar-track" style={{ height: BAR_AREA_HEIGHT }}>
                <div
                  className={b.band_min === null ? 'diff-hist-bar unscored' : 'diff-hist-bar'}
                  style={{ height: heightPx }}
                />
              </div>
              <span className="diff-hist-label">{b.label}</span>
            </button>
          )
        })}
      </div>
      {hint && <p className="diff-hist-hint muted">{hint}</p>}
    </div>
  )
}

export default DifficultyHistogram
