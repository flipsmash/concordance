import { useEffect, useRef } from 'react'

// Ctrl+S (⌘S on a Mac) presses a page's "Surprise me" button. The browser's
// own save-page shortcut is cancelled on those pages only; held keys don't
// auto-repeat it. `enabled=false` ignores the keys (e.g. while one is
// already loading).
export const SURPRISE_SHORTCUT_HINT = 'Surprise me (Ctrl+S / ⌘S)'

export default function useSurpriseShortcut(onSurprise, enabled = true) {
  const handler = useRef(onSurprise)
  handler.current = onSurprise

  useEffect(() => {
    function onKeyDown(e) {
      if (!(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey || e.key.toLowerCase() !== 's') return
      e.preventDefault()
      if (e.repeat || !enabled) return
      handler.current()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [enabled])
}
