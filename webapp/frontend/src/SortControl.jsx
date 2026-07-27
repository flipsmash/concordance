import './SortControl.css'

// Shared by Books and Authors -- a field picker plus a direction toggle.
// Clicking the toggle re-invokes onSort with the CURRENT field, which
// usePagedTable's handleSort already treats as "flip direction" (only a
// different field resets to ascending), so no extra state lives here.
function SortControl({ fields, sort, dir, onSort }) {
  return (
    <div className="sort-control">
      <label className="sort-control-label">
        Sort by
        <select value={sort} onChange={(e) => onSort(e.target.value)}>
          {fields.map((f) => (
            <option key={f.key} value={f.key}>{f.label}</option>
          ))}
        </select>
      </label>
      <button
        type="button"
        className="sort-dir-toggle"
        onClick={() => onSort(sort)}
        title={dir === 'asc' ? 'Ascending -- click for descending' : 'Descending -- click for ascending'}
      >
        {dir === 'asc' ? '↑' : '↓'}
      </button>
    </div>
  )
}

export default SortControl
