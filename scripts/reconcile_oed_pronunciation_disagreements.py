#!/usr/bin/env python3
"""Phase 1b: a genuine third vision-LLM pass for oed.entry rows where
pass1/pass2 (the original double-pass transcription) genuinely disagree and
_reconcile_close (the narrow, notation-only merge in
concordance/oed/pronunciation.py) can't safely resolve them either. Unlike
the double-pass design, this shows the model BOTH candidate readings
alongside the crop and asks it to adjudicate (pick A, pick B, or -- if both
are wrong -- transcribe it itself), rather than reading blind.

Measured in a 35-entry pilot: 100% resolution rate (model always reached a
verdict), ~2.8s/entry once the model is loaded (the first call in a fresh
process pays a one-time ~20-30s warmup, not a per-entry cost).

Deliberately narrower than "every stuck entry":
  - Only genuine pass1 != pass2 disagreements. A separate ~662-entry bucket
    where pass1 == pass2 but BOTH share the known leading-syllable-drop bias
    (caught by resolve_pronunciation's raw_ocr cross-check, e.g.
    "synclastic") is NOT handled here -- reusing the same model on the same
    image via an A/B prompt risks just re-confirming the same shared bias.
    Needs a differently-targeted prompt; left for later.
  - Only headwords matching a live concordance gap word (--all overrides
    this) -- that's the actual deliverable, the rest is optional cleanup.
  - Only entries whose source volume PDF is present in dictionaries/ --
    5 of 20 volumes were missing at time of writing (see --list-missing-volumes).
  - The SAME leading-char cross-check resolve_pronunciation always applies
    is applied here too, to whatever the reconciliation produces -- a
    verdict that fails it is left needs_review=True rather than trusted.

Dry-run by default: prints counts + a sample of what WOULD change, no model
load, no DB writes. Pass --apply to actually run inference and write.

Usage:
    python scripts/reconcile_oed_pronunciation_disagreements.py                 # dry run, prioritized subset
    python scripts/reconcile_oed_pronunciation_disagreements.py --apply
    python scripts/reconcile_oed_pronunciation_disagreements.py --apply --all   # every disagreement, not just concordance-matched
    python scripts/reconcile_oed_pronunciation_disagreements.py --list-missing-volumes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # noqa: E402
from rich.console import Console  # noqa: E402

from concordance import db  # noqa: E402
from concordance.oed import db as oed_db  # noqa: E402
from concordance.oed import pronunciation as p  # noqa: E402
from concordance.oed import segment  # noqa: E402
from concordance.oed.config import OedConfig  # noqa: E402

console = Console()

RECONCILE_PROMPT_TMPL = (
    "This image shows one cropped entry from a scanned Oxford English "
    "Dictionary page. The bold headword is {headword!r}, followed by a "
    "pronunciation in parentheses using IPA-like phonetic notation.\n\n"
    "Two earlier automated readings of the pronunciation disagreed:\n"
    "  Reading A: {pass1}\n"
    "  Reading B: {pass2}\n\n"
    "Look carefully at the actual pronunciation in the image and decide: "
    "is Reading A correct, is Reading B correct, or are both wrong (in "
    "which case transcribe it yourself)? Output ONLY a JSON object: "
    '{{"correct": "A"}} or {{"correct": "B"}} or {{"correct": "other", '
    '"pronunciation": "..."}}. No prose, no code fences.'
)


def fetch_candidates(conn, schema: str, all_entries: bool, present_volumes: set[str]) -> list[dict]:
    cur = conn.cursor()
    join = "" if all_entries else f"JOIN {db._safe_schema(db.DEFAULT_SCHEMA)}.word w ON w.lemma = e.headword_norm AND w.active AND (w.ipa IS NULL OR w.ipa = '')"
    cur.execute(f"""
        SELECT DISTINCT e.id, e.headword, v.file_name, e.page_number,
               e.pronunciation_pass1, e.pronunciation_pass2, e.pronunciation_raw
        FROM {schema}.entry e
        JOIN {schema}.volume v ON v.id = e.volume_id
        {join}
        WHERE e.pronunciation_pass1 IS NOT NULL AND e.pronunciation_pass2 IS NOT NULL
          AND (e.pronunciation_ipa IS NULL OR e.pronunciation_ipa = '')
    """)
    rows = cur.fetchall()
    out = []
    for eid, hw, fname, pg, pass1, pass2, raw in rows:
        if fname not in present_volumes:
            continue
        n1 = pass1.strip().strip("()").strip()
        n2 = pass2.strip().strip("()").strip()
        if n1 == n2:
            continue  # the cross-check-rejected-despite-agreement bucket -- not handled here
        out.append({"id": eid, "headword": hw, "file_name": fname, "page_number": pg,
                     "pass1": pass1, "pass2": pass2, "raw": raw})
    return out


def parse_verdict(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    try:
        v = json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw": text}
    if not isinstance(v, dict):
        return {"_parse_error": True, "_raw": text}
    return v


def verdict_to_ipa(entry: dict, verdict: dict) -> str | None:
    correct = verdict.get("correct")
    if correct == "A":
        return entry["pass1"].strip().strip("()").strip()
    if correct == "B":
        return entry["pass2"].strip().strip("()").strip()
    if correct == "other":
        candidate = verdict.get("pronunciation")
        if isinstance(candidate, str):
            return candidate.strip().strip("()").strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="oed")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--all", action="store_true", help="Every disagreement, not just concordance-matched headwords.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--commit-every", type=int, default=25)
    parser.add_argument("--list-missing-volumes", action="store_true")
    args = parser.parse_args()

    conn = db.connect(args.database_url)
    schema = db._safe_schema(args.schema)

    dictionaries_dir = Path("dictionaries")
    present_volumes = {f.name for f in dictionaries_dir.glob("*.pdf")}

    if args.list_missing_volumes:
        cur = conn.cursor()
        cur.execute(f"SELECT file_name FROM {schema}.volume ORDER BY id")
        for (fname,) in cur.fetchall():
            if fname not in present_volumes:
                console.print(f"[red]missing[/red]: {fname}")
        return

    candidates = fetch_candidates(conn, schema, args.all, present_volumes)
    console.print(f"[bold]{len(candidates)}[/bold] genuine-disagreement candidates "
                  f"({'all entries' if args.all else 'concordance-matched'}, on-disk volumes only)")
    if args.limit:
        candidates = candidates[:args.limit]

    if not args.apply:
        console.print("\n[yellow]dry run — pass --apply to run inference and write[/yellow]")
        for c in candidates[:10]:
            console.print(f"  {c['headword']!r}: {c['pass1']!r} / {c['pass2']!r}")
        return

    console.print("loading vision model...")
    t0 = time.time()
    cfg = OedConfig()
    transcriber_cfg_ok = Path(cfg.vision_model_path).exists() and Path(cfg.vision_mmproj_path).exists()
    if not transcriber_cfg_ok:
        console.print("[red]✗[/red] vision model files not found; nothing to run")
        return
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    handler = Qwen25VLChatHandler(clip_model_path=cfg.vision_mmproj_path, verbose=False)
    llm = Llama(model_path=cfg.vision_model_path, chat_handler=handler,
                n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)
    console.print(f"model loaded in {time.time() - t0:.1f}s")

    docs: dict[str, fitz.Document] = {}

    def get_doc(file_name: str) -> fitz.Document:
        if file_name not in docs:
            docs[file_name] = fitz.open(dictionaries_dir / file_name)
        return docs[file_name]

    stats = {"picked_A": 0, "picked_B": 0, "other": 0, "crop_not_found": 0,
             "unparseable": 0, "invalid_ipa": 0, "leading_char_crosscheck_failed": 0, "written": 0}

    cur = conn.cursor()
    t_start = time.time()
    for i, entry in enumerate(candidates, 1):
        doc = get_doc(entry["file_name"])
        page = doc[entry["page_number"]]
        hits = segment.find_headwords(page, cfg)
        target = entry["headword"].strip().lower()
        match = next((h for h in hits if h["text"].strip().lower() == target), None)
        if match is None:
            match = next((h for h in hits if h["text"].strip().lower().startswith(target[:6])), None)
        if match is None:
            stats["crop_not_found"] += 1
            continue
        rect = p.crop_rect(match["bbox"], page.rect, cfg)
        crop = p.render_crop(page, rect, cfg)

        prompt = RECONCILE_PROMPT_TMPL.format(headword=entry["headword"], pass1=entry["pass1"], pass2=entry["pass2"])
        data_uri = p._to_data_uri(crop)
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ]}],
            temperature=0.0, max_tokens=100,
        )
        verdict = parse_verdict(out["choices"][0]["message"]["content"])
        if verdict.get("_parse_error"):
            stats["unparseable"] += 1
            continue

        ipa = verdict_to_ipa(entry, verdict)
        if ipa is None or not p.valid_ipa(ipa):
            stats["invalid_ipa"] += 1
            continue

        raw_says = p._raw_suggests_leading_char(entry["raw"])
        ipa_says = p._ipa_has_leading_char(ipa)
        if raw_says is True and ipa_says is False:
            stats["leading_char_crosscheck_failed"] += 1
            continue

        key = {"A": "picked_A", "B": "picked_B"}.get(verdict.get("correct"), "other")
        stats[key] += 1

        oed_db.update_pronunciation(
            conn, entry["id"], pronunciation_raw=entry["raw"], pass1=entry["pass1"], pass2=entry["pass2"],
            ipa=ipa, source="vision_llm", needs_review=False, schema=schema,
        )
        stats["written"] += 1

        if i % args.commit_every == 0:
            conn.commit()
            elapsed = time.time() - t_start
            rate = elapsed / i
            remaining = (len(candidates) - i) * rate
            console.print(f"  ...{i}/{len(candidates)} ({rate:.1f}s/entry avg, "
                           f"~{remaining/60:.0f}min remaining) — {dict(stats)}")

    conn.commit()
    console.print(f"\n[green]✓[/green] {stats['written']} written in {time.time()-t_start:.0f}s — {dict(stats)}")


if __name__ == "__main__":
    main()
