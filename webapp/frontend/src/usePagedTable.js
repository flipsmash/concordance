import { useCallback, useEffect, useState } from 'react'

const API_BASE = '' // relative — dev proxies /api via vite.config.js, prod is same-origin

/** Shared pagination/sort/fetch lifecycle for a sortable, paginated API table. */
export function usePagedTable({ endpoint, pageSize = 50, defaultSort, defaultDir = 'asc', extraParams = {} }) {
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [sort, setSort] = useState(defaultSort)
  const [dir, setDir] = useState(defaultDir)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const extraKey = JSON.stringify(extraParams)

  const load = useCallback(() => {
    setLoading(true)
    setError(null)
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize), sort, dir })
    for (const [key, value] of Object.entries(extraParams)) {
      if (Array.isArray(value)) {
        value.forEach((v) => params.append(key, v))
      } else if (value) {
        params.set(key, value)
      }
    }
    fetch(`${API_BASE}${endpoint}?${params}`)
      .then((res) => {
        if (!res.ok) throw new Error(`load failed: ${res.status}`)
        return res.json()
      })
      .then((data) => {
        setItems(data.items)
        setTotal(data.total)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
    // extraParams is reduced to a stable string key below; the object itself
    // is recreated every render by callers passing an inline literal.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endpoint, page, pageSize, sort, dir, extraKey])

  useEffect(() => {
    load()
  }, [load])

  function handleSort(key) {
    if (sort === key) {
      setDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSort(key)
      setDir('asc')
    }
    setPage(1)
  }

  function resetPage() {
    setPage(1)
  }

  return {
    items,
    setItems,
    total,
    setTotal,
    page,
    setPage,
    sort,
    dir,
    handleSort,
    loading,
    error,
    setError,
    load,
    resetPage,
    totalPages: Math.max(1, Math.ceil(total / pageSize)),
  }
}
