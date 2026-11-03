"""Page-level span extraction — the shared geometry primitives segment.py and
parse.py build on. See the pilot notes in pronunciation.py's module docstring
for why this operates on span bounding boxes rather than trusting PyMuPDF's
own line/block grouping (its line breaks split same-line content unpredictably
on these scans, confirmed against real entries during the pilot)."""

from __future__ import annotations

import collections
import statistics

import fitz
import numpy as np


def flatten_spans(page: fitz.Page) -> list[dict]:
    d = page.get_text("dict")
    spans = []
    for block in d["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for s in line["spans"]:
                if s["text"].strip():
                    spans.append(s)
    return spans


def body_size(spans: list[dict]) -> float | None:
    """Char-weighted median span size — the page's running-prose baseline,
    used as the reference point for 'this is headword-sized' decisions."""
    sizes: collections.Counter = collections.Counter()
    for s in spans:
        sizes[round(s["size"], 1)] += len(s["text"])
    if not sizes:
        return None
    total = sum(sizes.values())
    acc = 0
    for sz in sorted(sizes):
        acc += sizes[sz]
        if acc >= total / 2:
            return sz
    return None


def group_rows(spans: list[dict], y_tol: float = 1.5) -> list[dict]:
    """Cluster spans into visual rows by vertical-center proximity — NOT by
    PyMuPDF's own 'line' grouping, which the pilot found splits same-line
    content (headword + pronunciation bracket) into separate line objects on
    a meaningful fraction of real entries.

    Spans are processed in yc-sorted order, so a span can only extend one of
    the last few rows created (anything older is already outside y_tol) —
    checking a small trailing window instead of every row turns this from
    O(n^2) into O(n), which matters at ~5k spans/page (a dense OED page,
    measured against real Volume I pages)."""
    ordered = sorted(spans, key=lambda s: (s["bbox"][1] + s["bbox"][3]) / 2)
    rows: list[dict] = []
    window = 6  # a couple of interleaved two-column rows' worth of slack
    for s in ordered:
        yc = (s["bbox"][1] + s["bbox"][3]) / 2
        for row in rows[-window:]:
            if abs(yc - row["yc"]) <= y_tol:
                row["spans"].append(s)
                row["yc"] = (row["yc"] * row["n"] + yc) / (row["n"] + 1)
                row["n"] += 1
                break
        else:
            rows.append({"yc": yc, "n": 1, "spans": [s]})
    return rows


def detect_column_bounds(spans: list[dict], page_width: float, min_gap: float = 8.0) -> list[tuple[float, float]]:
    """Column x-ranges via gutter detection — merge every span's horizontal
    extent into a coverage map, then split on gaps wide enough to be a real
    column gutter (not just ordinary inter-word spacing).

    Deliberately NOT a fixed 2-column assumption: a real Volume I page (80)
    measured 3 columns (x0 histogram showed clean gaps at ~205pt and ~400pt
    on a 629pt-wide page) — the OED's layout isn't uniformly 2-column
    throughout, and hardcoding the count silently merged columns 2 and 3
    into one row-grouping pass, which spliced unrelated entries' text
    together (confirmed: "abaft"'s raw_text picked up "abalienate"'s
    etymology because both had spans landing on the same clustered row)."""
    if not spans:
        return [(0.0, page_width)]
    intervals = sorted((s["bbox"][0], s["bbox"][2]) for s in spans)
    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    bounds = []
    start = 0.0
    for i in range(len(merged) - 1):
        gap_start, gap_end = merged[i][1], merged[i + 1][0]
        if gap_end - gap_start >= min_gap:
            bounds.append((start, gap_start))
            start = gap_end
    bounds.append((start, page_width))
    return bounds


def partition_by_column(spans: list[dict], bounds: list[tuple[float, float]]) -> list[list[dict]]:
    columns: list[list[dict]] = [[] for _ in bounds]
    for s in spans:
        xc = (s["bbox"][0] + s["bbox"][2]) / 2
        idx = len(bounds) - 1
        for i, (b0, b1) in enumerate(bounds):
            if b0 <= xc < b1:
                idx = i
                break
        columns[idx].append(s)
    return columns


def column_margin(rows: list[dict]) -> float | None:
    """A single column's left margin — the most common row-leading x0
    within that column's own rows (rows must already be column-partitioned;
    computing this across a whole unpartitioned page conflates columns)."""
    counts: collections.Counter = collections.Counter()
    for row in rows:
        if not row["spans"]:
            continue
        first = min(row["spans"], key=lambda s: s["bbox"][0])
        counts[round(first["bbox"][0])] += 1
    if not counts:
        return None
    merged: list[list[float]] = []
    for x, n in sorted(counts.items()):
        if merged and x - merged[-1][-1] <= 3:
            merged[-1].extend([x] * n)
        else:
            merged.append([x] * n)
    best = max(merged, key=len)
    return sum(best) / len(best)


def page_columns(spans: list[dict], page_width: float) -> tuple[list[list[dict]], list[float]]:
    """Partition a page's spans into columns, group rows WITHIN each column
    (never across), and return (rows_in_reading_order_per_column, margins)
    — margins[i] is columns[i]'s own left edge, aligned by index with the
    row lists. Reading order overall is columns[0]'s rows top-to-bottom,
    then columns[1]'s, etc. — concatenate the returned lists in order."""
    bounds = detect_column_bounds(spans, page_width)
    col_spans = partition_by_column(spans, bounds)
    all_col_rows = [sorted(group_rows(cs), key=lambda r: r["yc"]) for cs in col_spans]
    # Keep col_rows/margins aligned by index: an empty column (no spans
    # landed in that gutter-bounded x-range) has no margin, and dropping it
    # from `margins` alone while keeping it in `col_rows` would silently
    # misalign every column after it under zip() in segment.find_headwords.
    col_rows, margins = [], []
    for rows in all_col_rows:
        m = column_margin(rows)
        if m is not None:
            col_rows.append(rows)
            margins.append(m)
    return col_rows, margins


def row_text(row: dict) -> str:
    return " ".join(s["text"] for s in sorted(row["spans"], key=lambda s: s["bbox"][0]))


# --- bold, read from the scan -------------------------------------------------
#
# The text layer reports every span as the same font and flags, so it can't
# say what's bold -- but the pixels can. A word's stroke width (its mean run of
# dark pixels, left to right) divided by the page's median is measured at 1.5+
# for headwords and under ~1.3 for ordinary flush-left text.

INK_DARK_LEVEL = 110        # 0-255 grey, on a flattened page, below which a pixel counts as ink
BACKGROUND_BLOCK_PT = 12    # cell for estimating the paper: wider than a stroke, narrower than the shadow
BACKGROUND_PERCENTILE = 95
BACKGROUND_FLOOR = 40       # never scale up what is black in itself (a page edge)


def flatten_background(gray: np.ndarray, block: int) -> np.ndarray:
    """`gray` rescaled so the local paper is white. Scans differ in exposure and
    darken towards the spine, so one fixed ink level can sit at the ink's own
    grey on one volume and read paper as ink in a shadow on another. The paper
    in each `block`-square cell is that cell's bright end."""
    h, w = gray.shape
    rows, cols = max(h // block, 1), max(w // block, 1)
    h -= h % rows                       # drop the remainder so the cells are equal; it reads as the last cell
    w -= w % cols
    cells = gray[:h, :w].reshape(rows, h // rows, cols, w // cols)
    paper = np.percentile(cells, BACKGROUND_PERCENTILE, axis=(1, 3))
    paper = np.maximum(paper, BACKGROUND_FLOOR)
    row_of = np.minimum(np.arange(gray.shape[0]) // (h // rows), rows - 1)
    col_of = np.minimum(np.arange(gray.shape[1]) // (w // cols), cols - 1)
    return np.clip(gray / paper[np.ix_(row_of, col_of)] * 255, 0, 255).astype(np.uint8)


def page_gray(page: fitz.Page, zoom: float) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    return flatten_background(gray, int(BACKGROUND_BLOCK_PT * zoom))


def stroke_width(gray: np.ndarray, bbox, zoom: float) -> float | None:
    """Mean horizontal dark-run length (px) inside `bbox` (PDF points), or
    None if the box holds no ink."""
    x0, y0, x1, y1 = (int(v * zoom) for v in bbox)
    crop = gray[max(y0, 0):y1, max(x0, 0):x1]
    if crop.size == 0:
        return None
    dark = crop < INK_DARK_LEVEL
    runs = (np.diff(dark.astype(np.int8), axis=1) == 1).sum()
    return float(dark.sum() / runs) if runs else None


def median_stroke(gray: np.ndarray, spans: list[dict], zoom: float) -> float | None:
    """The page's typical stroke width, from words long enough to measure."""
    widths = [stroke_width(gray, s["bbox"], zoom) for s in spans if len(s["text"].strip()) >= 4]
    widths = [w for w in widths if w]
    return statistics.median(widths) if widths else None


def row_top(row: dict) -> float:
    return min(s["bbox"][1] for s in row["spans"])
