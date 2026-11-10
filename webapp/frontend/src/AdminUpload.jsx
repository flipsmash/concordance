import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import './AdminUpload.css'

const API_BASE = ''

// Admin: upload a book (.txt / .pdf / .epub) for the corpus. The server
// stores it and starts `concordance process-uploads` in the background:
// ingest, then the upkeep (maintain, genres, fame, stats), then the
// relatedness rebuild -- the list below follows each upload through.

const STATUS = {
  queued: 'Waiting',
  ingesting: 'Importing',
  in_corpus: 'In the collection',
  upkeep: 'In the collection · updating',
  done: 'Done',
  failed: 'Failed',
}

const STEP = {
  ingest: 'reading the book and picking out its words',
  maintain: 'definitions, categories, difficulty, quiz definitions…',
  'book-genres': 'tagging genres',
  'author-fame': 'scoring author fame',
  'book-fame': 'scoring book fame',
  'author-stats': 'author statistics',
  'book-stats': 'book statistics',
  'refresh-rejected-index': 'rejected-word list',
  relate: 'rebuilding book & author relatedness (about an hour)',
  'corpus-map': 'redrawing the full-collection map',
  'force-map': 'redrawing the force maps of every book, author and word',
}

// "[Title] -- [Author].epub", the incoming/ folder's naming convention
function guessFromName(name) {
  const stem = name.replace(/\.[^.]+$/, '').replace(/_/g, ' ')
  const [title, author] = stem.split(/\s+--\s+/)
  return { title: (title || '').trim(), author: (author || '').trim() }
}

function AdminUpload() {
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [author, setAuthor] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [existing, setExisting] = useState(null) // a book with this title already
  const [uploads, setUploads] = useState(null)

  const load = useCallback(() => {
    fetch(`${API_BASE}/api/admin/uploads`)
      .then((res) => (res.ok ? res.json() : []))
      .then(setUploads)
      .catch(() => setUploads([]))
  }, [])

  useEffect(load, [load])
  const busy = (uploads || []).some((u) => u.status !== 'done' && u.status !== 'failed')
  useEffect(() => {
    if (!busy) return undefined
    const t = setInterval(load, 10000)
    return () => clearInterval(t)
  }, [busy, load])

  function pick(f) {
    setFile(f)
    setError('')
    setExisting(null)
    if (f) {
      const g = guessFromName(f.name)
      setTitle(g.title)
      setAuthor(g.author)
    }
  }

  function send(replace = false) {
    if (!file || !title.trim()) return
    const body = new FormData()
    body.append('file', file)
    body.append('title', title)
    body.append('author', author)
    body.append('replace', replace ? 'true' : 'false')
    setSending(true)
    setError('')
    fetch(`${API_BASE}/api/admin/uploads`, { method: 'POST', body })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}))
        if (res.status === 409 && data.detail?.existing) {
          setExisting(data.detail.existing)
          return
        }
        if (!res.ok) {
          const d = data.detail
          throw new Error(typeof d === 'string' ? d : d?.message || `upload failed (${res.status})`)
        }
        setFile(null)
        setTitle('')
        setAuthor('')
        setExisting(null)
        document.getElementById('upload-file').value = ''
        load()
      })
      .catch((err) => setError(err.message || 'upload failed'))
      .finally(() => setSending(false))
  }

  return (
    <div className="admin-upload">
      <h2>Upload a book</h2>
      <p className="muted">
        A .txt, .pdf (with real text, not a scan) or .epub file, up to 90 MB. It is imported in the background: its
        words appear within minutes, and the rest &mdash; categories, difficulty, genres, fame, and finally book
        and author relatedness &mdash; follows over the next hour or so.
      </p>

      <form className="admin-upload-form" onSubmit={(e) => { e.preventDefault(); send(false) }}>
        <label>
          File
          <input id="upload-file" type="file" accept=".txt,.pdf,.epub"
            onChange={(e) => pick(e.target.files?.[0] || null)} disabled={sending} />
        </label>
        <label>
          Title
          <input value={title} onChange={(e) => { setTitle(e.target.value); setExisting(null) }}
            disabled={sending} maxLength={300} required />
        </label>
        <label>
          Author
          <input value={author} onChange={(e) => setAuthor(e.target.value)} disabled={sending}
            placeholder="Last, First — as the other books have it" />
        </label>
        <div>
          <button type="submit" className="accept-btn" disabled={sending || !file || !title.trim() || !!existing}>
            {sending ? 'Uploading…' : 'Upload'}
          </button>
        </div>
      </form>

      {existing && (
        <div className="admin-upload-confirm">
          <p>
            <strong>
              <Link to={`/app/books?q=${encodeURIComponent(existing.title)}`}>{existing.title}</Link>
            </strong>
            {existing.author && <> by {existing.author}</>} is already in the collection. Importing this file
            updates that book in place: its words are re-read from this file.
          </p>
          <button type="button" onClick={() => setExisting(null)} disabled={sending}>Cancel</button>{' '}
          <button type="button" className="accept-btn" onClick={() => send(true)} disabled={sending}>
            Update the existing book
          </button>
        </div>
      )}
      {error && <div className="error-banner">{error}</div>}

      <h3>Recent uploads</h3>
      {uploads === null ? (
        <p className="muted">Loading…</p>
      ) : uploads.length === 0 ? (
        <p className="muted">None yet.</p>
      ) : (
        <ul className="admin-upload-list">
          {uploads.map((u) => (
            <li key={u.id} className={`admin-upload-item status-${u.status}`}>
              <div className="admin-upload-head">
                <strong>{u.title}</strong>
                {u.author && <span className="muted"> — {u.author}</span>}
                <span className="admin-upload-status">{STATUS[u.status] || u.status}</span>
              </div>
              <div className="muted admin-upload-meta">
                {u.filename} · uploaded {new Date(u.uploaded_at).toLocaleString()} by {u.uploaded_by}
                {u.words_kept != null && <> · {u.words_kept.toLocaleString()} words kept, {u.words_rejected.toLocaleString()} rejected</>}
              </div>
              {u.step && <div className="admin-upload-step">Now: {STEP[u.step] || u.step}</div>}
              {u.book_id && u.status !== 'failed' && (
                <Link className="admin-upload-link"
                  to={u.author ? `/app/authors/${encodeURIComponent(u.author)}/${u.book_id}` : `/app/books?q=${encodeURIComponent(u.title)}`}>
                  Open the book →
                </Link>
              )}
              {u.error && (
                <details className="admin-upload-error">
                  <summary>{u.status === 'failed' ? 'What went wrong' : 'Some upkeep steps had problems'}</summary>
                  <pre>{u.error}</pre>
                </details>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default AdminUpload
