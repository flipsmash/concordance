import { useState } from 'react'

const API_BASE = ''

// Admin-only: generates a one-time /register?token=... link. Reachable only
// because it's rendered inside AcceptedView, which sits behind Cloudflare
// Access -- see require_admin in webapp/backend/main.py for the backend side.
function InviteButton() {
  const [open, setOpen] = useState(false)
  const [link, setLink] = useState('')
  const [expiresAt, setExpiresAt] = useState('')
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState('')

  function generate() {
    setError('')
    fetch(`${API_BASE}/api/admin/invites`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`request failed (${res.status})`)
        return res.json()
      })
      .then((data) => {
        setLink(`${window.location.origin}${data.register_path}`)
        setExpiresAt(new Date(data.expires_at).toLocaleString())
        setOpen(true)
        setCopied(false)
      })
      .catch((err) => setError(err.message || 'failed to generate invite'))
  }

  function copy() {
    navigator.clipboard.writeText(link).then(() => setCopied(true))
  }

  return (
    <div className="invite-button-wrap">
      <button type="button" className="invite-btn" onClick={generate}>
        + Generate Invite Link
      </button>
      {error && <span className="invite-error">{error}</span>}
      {open && (
        <div className="invite-panel">
          <input type="text" readOnly value={link} onFocus={(e) => e.target.select()} />
          <button type="button" onClick={copy}>
            {copied ? 'Copied' : 'Copy'}
          </button>
          <span className="invite-expiry">expires {expiresAt}</span>
          <button type="button" className="invite-dismiss" onClick={() => setOpen(false)}>
            ×
          </button>
        </div>
      )}
    </div>
  )
}

export default InviteButton
