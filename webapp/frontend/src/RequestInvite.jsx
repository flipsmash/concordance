import { useState } from 'react'
import { Link } from 'react-router-dom'
import { LogoMark } from './Logo'
import './Auth.css'

const REQUEST_ADDRESS = 'concordance.vocab@gmail.com'

// No backend involved -- this app has no email-sending infrastructure (see
// /api/auth/register's own docstring). A real <a href="mailto:..."> -- no
// onClick, no JS in the path at all, just a link with a real href.
function RequestInvite() {
  const [email, setEmail] = useState('')

  const subject = encodeURIComponent('Account Request')
  const body = encodeURIComponent(email ? `Please send an invite link to: ${email}` : '')
  const mailtoHref = `mailto:${REQUEST_ADDRESS}?subject=${subject}&body=${body}`

  return (
    <div className="auth-page">
      <div className="auth-form">
        <div className="auth-brand"><LogoMark width={170} /></div>
        <h1>Request an invite</h1>
        <p className="auth-hint">
          This app is invite-only. Enter the email you&apos;d like the invite sent to, then click the
          link below.
        </p>
        <label>
          Your email
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoFocus />
        </label>
        <a className="auth-cta" href={mailtoHref}>
          Email request
        </a>
        <p className="auth-hint">
          <Link to="/login">← Back to log in</Link>
        </p>
      </div>
    </div>
  )
}

export default RequestInvite
