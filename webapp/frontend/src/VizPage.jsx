import { Link } from 'react-router-dom'
import './VizPage.css'

// The one header every visualization page uses: a way back to the gallery,
// the title in the display face (as on Home), a short plain-language lede,
// and optional controls on the right. Pages used to each roll their own
// (serif here, sans there, a back link top-left or top-right, or none).
export function VizPageHeader({ title, children, actions = null, back = true }) {
  return (
    <div className="viz-page-header">
      <div className="viz-page-heading">
        {back && <Link to="/app/visualizations" className="viz-page-crumb">← Visualizations</Link>}
        <h1 className="viz-page-title">{title}</h1>
        {children && <p className="viz-page-lede">{children}</p>}
      </div>
      {actions && <div className="viz-page-actions">{actions}</div>}
    </div>
  )
}

// Standard page frame: centred column with the shared gutters.
export function VizPage({ wide = false, children }) {
  return <div className={`viz-page${wide ? ' wide' : ''}`}>{children}</div>
}
