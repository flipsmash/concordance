import { useEffect, useState } from 'react'
import GrowthChart from './GrowthChart'
import { VizPage, VizPageHeader } from './VizPage'
import './Browse.css' // .browse-pill
import './Visualizations.css' // .growth-chart-*

const API_BASE = ''

// Every calendar day from the earliest to the latest date across ALL the
// series, so Books / Authors / Words share one timeline.
function sharedDays(...series) {
  const dates = series.flatMap((s) => (s || []).map((d) => d.date)).sort()
  if (!dates.length) return []
  const out = []
  const end = new Date(`${dates[dates.length - 1]}T00:00:00Z`)
  for (let d = new Date(`${dates[0]}T00:00:00Z`); d <= end; d.setUTCDate(d.getUTCDate() + 1)) {
    out.push(d.toISOString().slice(0, 10))
  }
  return out
}

// {date, count} per day -> {date, value}: weekly totals (ingests land in
// occasional batch passes, so daily counts were all spikes), or the running
// total.
function toGrowthSeries(daily, mode, days) {
  if (daily === null) return null // still loading
  const byDate = new Map(daily.map((d) => [d.date, d.count]))
  if (mode === 'cumulative') {
    let running = 0
    return days.map((date) => {
      running += byDate.get(date) || 0
      return { date, value: running }
    })
  }
  const weeks = []
  days.forEach((date, i) => {
    if (i % 7 === 0) weeks.push({ date, value: 0 })
    weeks[weeks.length - 1].value += byDate.get(date) || 0
  })
  return weeks
}

function useSeries(metric) {
  const [data, setData] = useState(null)
  useEffect(() => {
    fetch(`${API_BASE}/api/browse/growth?metric=${metric}`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData([]))
  }, [metric])
  return data
}

function VizGrowth() {
  const [mode, setMode] = useState('weekly')
  const books = useSeries('books')
  const authors = useSeries('authors')
  const words = useSeries('words')
  const days = sharedDays(books, authors, words)
  const bars = mode !== 'cumulative'
  return (
    <VizPage>
      <VizPageHeader
        title="Corpus growth"
        actions={
          <div className="growth-chart-toggle">
            {[['weekly', 'Per week'], ['cumulative', 'Running total']].map(([id, label]) => (
              <button key={id} type="button" className={mode === id ? 'browse-pill active' : 'browse-pill'} onClick={() => setMode(id)}>
                {label}
              </button>
            ))}
          </div>
        }
      >
        Books, authors and words added to the collection over its whole history. Books arrive in occasional batch
        runs, so uneven weeks are real. &ldquo;Authors&rdquo; counts a real author&rsquo;s first book (labels like
        &ldquo;Various&rdquo; excluded); &ldquo;words&rdquo; counts each word&rsquo;s first appearance.
      </VizPageHeader>
      <div className="growth-chart-group">
        {[['Books', books, 'books'], ['Authors', authors, 'authors'], ['Words', words, 'words']].map(([label, series, unit]) => (
          <div className="growth-chart-row" key={label}>
            <span className="growth-chart-row-label">{label}</span>
            <GrowthChart data={toGrowthSeries(series, mode, days)} unit={unit} bars={bars} />
          </div>
        ))}
      </div>
    </VizPage>
  )
}

export default VizGrowth
