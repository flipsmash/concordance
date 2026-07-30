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
function DifficultyHistogram({ bands, selectedBand, onSelect }) {
  if (bands.length === 0) return null
  const maxCount = Math.max(...bands.map((b) => b.word_count), 1)

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
      <div className="diff-hist-bars">
        {bands.map((b) => {
          const heightPx = b.word_count === 0 ? 0 : Math.max((b.word_count / maxCount) * BAR_AREA_HEIGHT, 3)
          return (
            <button
              type="button"
              className={isSelected(b) ? 'diff-hist-col active' : 'diff-hist-col'}
              key={b.label}
              title={`${b.label}: ${b.word_count} words`}
              onClick={() => onSelect(b)}
            >
              <span className="diff-hist-count">{b.word_count > 0 ? b.word_count : ''}</span>
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
    </div>
  )
}

export default DifficultyHistogram
