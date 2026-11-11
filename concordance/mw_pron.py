"""Merriam-Webster pronunciations: real recordings and IPA.

`concordance mw-pronounce` works through the vocabulary worst-pronounced
first -- words whose audio is only a guess from the spelling (Piper), then
Azure's IPA-guided synthesis, then words with no audio -- and asks
Merriam-Webster for each:

  - the human recording, which replaces the synthesized clip (source 'mw',
    ranked with Commons as a real recording);
  - the respelling (\\ˈkän-ˌkȯr-dən(t)s\\), converted to IPA for a word with
    none (ipa_source 'mw').

It uses the licensed Collegiate API (mw.lookup_api): cached on disk, so a
word looked up before -- by ingest, mw-backfill or the audio step -- costs
nothing, and capped at 1,000 new lookups a day, of which this leaves a
reserve for ingest. Cached words go first, then new lookups until the
reserve. Each word is marked mw_pron_checked_at, hit or miss, so running it
once a day carries on where the last run stopped and picks up newly ingested
words.

Scraping the website instead (no daily cap) was asked for but isn't done:
the site sits behind a Cloudflare bot challenge, and getting past one was
ruled out. The API is the licensed route to the same recordings.
"""

from __future__ import annotations

import re
from pathlib import Path

from .db import _safe_schema

API_RESERVE = 100   # of the 1,000 daily lookups, left for ingest

# MW respelling -> IPA, longest first. MW marks stress before the syllable as
# IPA does; hyphens only divide syllables.
_MAP = [
    ("au̇", "aʊ"), ("ȯi", "ɔɪ"), ("u̇", "ʊ"), ("t͟h", "ð"),
    ("ch", "tʃ"), ("sh", "ʃ"), ("zh", "ʒ"), ("th", "θ"),
    ("ā", "eɪ"), ("ä", "ɑ"), ("a", "æ"), ("ē", "i"), ("e", "ɛ"), ("ī", "aɪ"), ("i", "ɪ"),
    ("ō", "oʊ"), ("ȯ", "ɔ"), ("ü", "u"), ("ə", "ə"), ("ᵊ", "ə"),
    ("j", "dʒ"), ("y", "j"), ("ŋ", "ŋ"), ("ˈ", "ˈ"), ("ˌ", "ˌ"),
] + [(c, c) for c in "bdfghklmnprstvwz"]
_FOREIGN = set("ⁿœᵫḵꟾ")


def mw_to_ipa(respelling: str) -> str | None:
    """IPA for one MW respelling, or None when it can't be trusted: a partial
    respelling (MW writes only the part that differs: "-ˈdäm"), foreign
    sounds (French nasal ⁿ, œ...), or a symbol this doesn't know."""
    text = (respelling or "").strip()
    text = re.split(r"[,;]", text)[0].strip()               # first of several variants
    if not text or text.startswith("-") or text.endswith("-") or set(text) & _FOREIGN:
        return None
    text = re.sub(r"\(\s*[ˈˌ]\s*\)", "", text)                # optional stress
    text = re.sub(r"\([^)]*\)", "", text)                     # optional sounds
    out = []
    for syllable in text.replace(" ", "-").split("-"):
        i = 0
        while i < len(syllable):
            for src, dst in _MAP:
                if syllable.startswith(src, i):
                    out.append(dst)
                    i += len(src)
                    break
            else:
                return None
    ipa = "".join(out)
    return ipa if re.search(r"[æɑɛɪiəʊuɔeoaʌ]", ipa) else None


def fetch_recording(url: str, dest: Path, tries: int = 3) -> bool:
    """Download an MW recording to `dest`. MW serves MP3 already, so no
    transcoding (audio.fetch_commons_audio converts through ffmpeg, which
    picks the output format from the file name). Written to a temporary
    file first, so a failed download never clobbers the clip it replaces."""
    import time

    import requests

    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "concordance-audio (personal vocab tool)"})
        except requests.RequestException:
            r = None
        # an MP3 starts with an ID3 tag or an MPEG frame sync byte
        if r is not None and r.status_code == 200 and (r.content[:3] == b"ID3" or r.content[:1] == b"\xff"):
            tmp = dest.with_name(f".{dest.name}.download")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return True
        if r is not None and r.status_code == 404:
            return False
        time.sleep(1 + attempt * 2)
    return False


def candidates(conn, schema: str) -> list[tuple[int, str, str | None, str | None, str | None]]:
    """Unchecked active words, worst pronunciation first: (id, lemma, ipa,
    part of speech, current audio source). Words with a real recording
    (Commons, MW) aren't candidates."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.lemma, CASE WHEN w.ipa_source = 'derived' THEN NULL ELSE w.ipa END,
                               w.part_of_speech, a.source
                        FROM {s}.word w LEFT JOIN {s}.word_audio a ON a.word_id = w.id
                        WHERE w.active AND w.mw_pron_checked_at IS NULL
                          AND coalesce(a.source, '') NOT IN ('commons', 'mw')
                        ORDER BY CASE a.source WHEN 'piper' THEN 0 WHEN 'azure_guess' THEN 1
                                               WHEN 'azure' THEN 2 ELSE 3 END, w.id""")
        return cur.fetchall()


def pronounce(conn, schema: str, *, limit: int = 0, reserve: int = API_RESERVE,
              audio_dir: Path | None = None, log=print) -> dict:
    """One run; see the module docstring. Returns counts."""
    from collections import Counter

    from . import audio, mw
    from .db.audio import _POS_TO_TAGGER
    from .dictionary import make_session

    s = _safe_schema(schema)
    audio_dir = audio_dir or audio.AUDIO_DIR
    key = mw.mw_api_key()
    if not key:
        log("no MW_DICTIONARY_API_KEY in .env")
        return {"checked": 0}
    rows = candidates(conn, schema)
    cache = mw.load_cache()
    cached = set(cache)
    # free first: words MW already answered for some other step
    rows.sort(key=lambda r: r[1].strip().lower() not in cached)
    session = make_session()
    stats: Counter = Counter(candidates=len(rows))
    audio_dir.mkdir(parents=True, exist_ok=True)

    with conn.cursor() as cur:
        for wid, lemma, ipa, pos, source in rows:
            if limit and stats["checked"] >= limit:
                break
            lemma_lc = lemma.strip().lower()
            if lemma_lc not in cached and mw.usage_today() >= mw.DAILY_QUOTA - reserve:
                stats["stopped_for_quota"] = 1
                break
            raw_entries = (mw.entries_from_raw(cache[lemma_lc]) if lemma_lc in cache
                           else mw.lookup_api(lemma, key, session))
            entries = mw.exact_matches(raw_entries, lemma)
            stats["checked"] += 1
            recorded = False
            if entries:
                entry = mw.pick_entry(entries, _POS_TO_TAGGER.get((pos or "").lower(), ""))
                prs = entry.pronunciations
                url = next((p.audio_url for p in prs if p.audio_url), None)
                if url:
                    dest = audio_dir / f"{lemma_lc}.mp3"
                    if fetch_recording(url, dest):
                        cur.execute(
                            f"""INSERT INTO {s}.word_audio (word_id, source, file_path, ipa_used, voice,
                                                            license_note, generated_at)
                                VALUES (%s, 'mw', %s, NULL, %s, %s, now())
                                ON CONFLICT (word_id) DO UPDATE SET source = 'mw',
                                    file_path = EXCLUDED.file_path, ipa_used = NULL, voice = EXCLUDED.voice,
                                    license_note = EXCLUDED.license_note, generated_at = now()""",
                            (wid, str(dest), url, "Merriam-Webster recording"))
                        stats["recordings"] += 1
                        recorded = True
                        stats[f"replaced_{source or 'none'}"] += 1
                if not (ipa or "").strip() or not audio.looks_like_english_ipa(ipa):
                    new = next((x for x in (mw_to_ipa(p.respelling) for p in prs) if x), None)
                    if new and audio.looks_like_english_ipa(new):
                        cur.execute(f"UPDATE {s}.word SET ipa = %s, ipa_source = 'mw', ipa_derived_from = NULL WHERE id = %s",
                                    (new, wid))
                        stats["ipa_added"] += 1
                        # no recording, but now IPA: voice that rather than keep the spelling guess
                        if not recorded and source in (None, "piper", "azure_guess", "piper_ipa"):
                            phonemes = audio.ipa_to_piper_phonemes(new)
                            clip = audio.synthesize_piper_ipa(phonemes) if phonemes else None
                            if clip:
                                dest = audio_dir / f"{lemma_lc}.mp3"
                                dest.write_bytes(clip)
                                cur.execute(
                                    f"""INSERT INTO {s}.word_audio (word_id, source, file_path, ipa_used, voice,
                                                                    license_note, generated_at)
                                        VALUES (%s, 'piper_ipa', %s, %s, %s, %s, now())
                                        ON CONFLICT (word_id) DO UPDATE SET source = 'piper_ipa',
                                            file_path = EXCLUDED.file_path, ipa_used = EXCLUDED.ipa_used,
                                            voice = EXCLUDED.voice, license_note = EXCLUDED.license_note,
                                            generated_at = now()""",
                                    (wid, str(dest), "".join(phonemes), audio.PIPER_IPA_VOICE_NAME,
                                     "Synthesized locally by Piper from the word's IPA"))
                                stats["voiced_from_ipa"] += 1
            else:
                stats["no_entry"] += 1
            cur.execute(f"UPDATE {s}.word SET mw_pron_checked_at = now() WHERE id = %s", (wid,))
            if stats["checked"] % 50 == 0:
                conn.commit()
                log(f"  {stats['checked']} checked, {stats['recordings']} recordings, {stats['ipa_added']} IPA")
    conn.commit()
    return dict(stats)
