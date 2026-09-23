"""Google Books Ngram features from the bulk v3 dataset (§ difficulty).

Replaces ngram.py's one-HTTP-request-per-word fetch for everything the bulk
data covers. Dataset: English 1-grams, v3 / "en-2019" (20200217) -- the same
corpus the Ngram Viewer JSON endpoint serves -- published as 24 hash-sharded
gzip files plus a per-year totals file:

    <term>\\t<year>,<match_count>,<volume_count>\\t<year>,...
    totalcounts-1: \\t<year>,<match_count>,<page_count>,<volume_count>...

Each term appears untagged (all occurrences) and again per part-of-speech tag
(`word_NOUN`); only the untagged line is kept, matching the Viewer's default
query. Terms are matched case-SENSITIVELY, lowercase, exactly as ngram.fetch
queried -- the difficulty scale's floor and its ~1:1 tracking of wordfreq
were calibrated on those values.

Flow: `download()` the shards once to a local cache (Linux filesystem, ~10
GB), `scan()` them into the `ngram.unigram` table (every lowercase word-like
term, not just the current vocabulary, so words added later never need a
re-scan), then `features()` turns a term's yearly counts into the same
{peak, recent, recency_ratio, peak_year} shape ngram.fetch returns.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

BASE_URL = "http://storage.googleapis.com/books/ngrams/books/20200217/eng"
N_SHARDS = 24
FIRST_YEAR, LAST_YEAR = 1500, 2019       # the range ngram.fetch asked the Viewer for
RECENT_YEARS = 20                        # 2000-2019, same as ngram._RECENT_YEARS
SMOOTHING = 3                            # Viewer's smoothing=3: mean over year±3

# Untagged, lowercase, word-shaped terms only (letters incl. UTF-8 accented,
# internal hyphen/apostrophe). grep runs byte-wise (LC_ALL=C), so the
# non-ASCII range is expressed as raw UTF-8 bytes.
_GREP_KEEP = r"^[a-z\x80-\xff][a-z'\x80-\xff-]*\t"
_TERM_RE = re.compile(r"[a-zÀ-ɏͰ-Ͽ][a-z'À-ɏͰ-Ͽ-]*")


def cache_dir() -> Path:
    return Path(os.environ.get("CONCORDANCE_NGRAM_DIR",
                               Path.home() / ".cache" / "concordance" / "ngram-v3"))


def shard_names() -> list[str]:
    return [f"1-{i:05d}-of-{N_SHARDS:05d}.gz" for i in range(N_SHARDS)]


def read_totals(path: Path) -> dict[int, int]:
    """year -> total 1-gram match count (the relative-frequency denominator)."""
    totals = {}
    for field in path.read_text().split():
        year, match, *_ = field.split(",")
        totals[int(year)] = int(match)
    return totals


def iter_shard(path: Path):
    """Yield (term, {year: match_count}) for every kept line of one shard.
    zcat|grep does the bulk filtering in C -- the shards are mostly numbers,
    tagged variants and punctuation."""
    zcat = subprocess.Popen(["zcat", str(path)], stdout=subprocess.PIPE)
    grep = subprocess.Popen(["grep", "-aP", _GREP_KEEP], stdin=zcat.stdout,
                            stdout=subprocess.PIPE, env={**os.environ, "LC_ALL": "C"})
    zcat.stdout.close()
    for raw in grep.stdout:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        term, _, rest = line.partition("\t")
        if not _TERM_RE.fullmatch(term):
            continue
        counts = {}
        for field in rest.split("\t"):
            year, match, _vol = field.split(",")
            counts[int(year)] = int(match)
        yield term, counts
    grep.wait(); zcat.wait()


def features(counts: dict[int, int], totals: dict[int, int]) -> dict:
    """ngram.fetch-compatible features from raw yearly counts: relative
    frequency per year over FIRST_YEAR..LAST_YEAR (0 where absent), smoothed
    the way the Viewer's smoothing=3 does (mean of the year and up to 3 on
    each side, window truncated at the range ends), then peak / peak year /
    2000-2019 mean / recent-over-peak."""
    years = range(FIRST_YEAR, LAST_YEAR + 1)
    rel = [(counts.get(y, 0) / totals[y]) if totals.get(y) else 0.0 for y in years]
    if not any(rel):
        return {"peak": 0.0, "recent": 0.0, "recency_ratio": None, "peak_year": None}
    n = len(rel)
    prefix = [0.0]
    for v in rel:
        prefix.append(prefix[-1] + v)
    smooth = []
    for i in range(n):
        lo, hi = max(0, i - SMOOTHING), min(n, i + SMOOTHING + 1)
        smooth.append((prefix[hi] - prefix[lo]) / (hi - lo))
    peak = max(smooth)
    peak_year = FIRST_YEAR + smooth.index(peak)
    recent = sum(smooth[-RECENT_YEARS:]) / RECENT_YEARS
    return {"peak": peak, "recent": recent,
            "recency_ratio": (recent / peak) if peak > 0 else None, "peak_year": peak_year}


def open_totals() -> dict[int, int]:
    return read_totals(cache_dir() / "totalcounts-1")


# --- decade bins (kept alongside the features for a future windowed
# decline signal, e.g. 1800-1899 vs 2000-2019 -- see archaic.py) ----------

FIRST_DECADE = 1800                      # pre-1800 is too small/OCR-noisy to bin usefully
DECADES = list(range(FIRST_DECADE, LAST_YEAR + 1, 10))   # 1800 .. 2010 (22 bins)


def decade_counts(counts: dict[int, int]) -> list[int]:
    bins = [0] * len(DECADES)
    for y, c in counts.items():
        if y >= FIRST_DECADE:
            bins[(y - FIRST_DECADE) // 10] += c
    return bins


def download(dest: Path | None = None, workers: int = 6) -> list[Path]:
    """Fetch totalcounts-1 and all shards into `dest` (resumable via curl -C -),
    then gzip-verify each so a truncated file fails loudly, not as missing words."""
    from concurrent.futures import ThreadPoolExecutor
    dest = dest or cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    names = ["totalcounts-1", *shard_names()]

    def verified(path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        return not path.name.endswith(".gz") or subprocess.run(
            ["gzip", "-t", str(path)], capture_output=True).returncode == 0

    def fetch(name: str) -> Path:
        out = dest / name
        if verified(out):               # curl -C - on a complete file is an HTTP 416 error
            return out
        subprocess.run(["curl", "-s", "--fail", "--retry", "5", "--retry-delay", "10", "-C", "-",
                        "-o", str(out), f"{BASE_URL}/{name}"], check=True)
        if name.endswith(".gz"):
            subprocess.run(["gzip", "-t", str(out)], check=True)
        return out

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fetch, names))


def _scan_one(args) -> tuple[str, int]:
    """Worker: one shard -> one TSV part (term, peak, recent, recency_ratio,
    peak_year, decade-count array literal)."""
    shard, part, totals = args
    n = 0
    with open(part, "w", encoding="utf-8") as out:
        for term, counts in iter_shard(shard):
            f = features(counts, totals)
            ratio = "" if f["recency_ratio"] is None else repr(f["recency_ratio"])
            year = "" if f["peak_year"] is None else str(f["peak_year"])
            bins = "{" + ",".join(map(str, decade_counts(counts))) + "}"
            out.write(f"{term}\t{f['peak']!r}\t{f['recent']!r}\t{ratio}\t{year}\t{bins}\n")
            n += 1
    return str(shard), n


def build_tsv(out_dir: Path | None = None, workers: int = 8) -> list[Path]:
    """Scan every shard in parallel into per-shard TSV parts ready for COPY."""
    from multiprocessing import Pool
    src = cache_dir()
    out_dir = out_dir or src / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = open_totals()
    jobs = [(src / name, out_dir / (name + ".tsv"), totals) for name in shard_names()]
    missing = [str(s) for s, _, _ in jobs if not s.exists()]
    if missing:
        raise FileNotFoundError(f"shards not downloaded yet: {missing[:3]}... run download() first")
    with Pool(workers) as pool:
        for shard, n in pool.imap_unordered(_scan_one, jobs):
            print(f"  {Path(shard).name}: {n:,} terms")
    return [part for _, part, _ in jobs]
