import './DifficultyHistogram.css'

const BAR_AREA_HEIGHT = 120 // px

// An actual histogram: bar HEIGHT encodes frequency (word_count), on a
// shared baseline, with equal-width bins along the x-axis -- not the main
// Browse page's proportional-WIDTH filter strip (a different, deliberately
// compact widget for a different job: click-to-set-a-range inline in a
// facet row). That one stays as-is; this is its own component because
// reusing it here rendered every band the same height, which isn't a
// histogram at all, just a segmented bar.
function DifficultyHistogram({ bands }) {
  if (bands.length === 0) return null
  const maxCount = Math.max(...bands.map((b) => b.word_count), 1)

  return (
    <div className="diff-hist">
      <div className="diff-hist-bars">
        {bands.map((b) => {
          const heightPx = b.word_count === 0 ? 0 : Math.max((b.word_count / maxCount) * BAR_AREA_HEIGHT, 3)
          return (
            <div className="diff-hist-col" key={b.label} title={`${b.label}: ${b.word_count} words`}>
              <span className="diff-hist-count">{b.word_count > 0 ? b.word_count : ''}</span>
              <div className="diff-hist-bar-track" style={{ height: BAR_AREA_HEIGHT }}>
                <div
                  className={b.band_min === null ? 'diff-hist-bar unscored' : 'diff-hist-bar'}
                  style={{ height: heightPx }}
                />
              </div>
              <span className="diff-hist-label">{b.label}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default DifficultyHistogram
