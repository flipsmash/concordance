"""dictionary.com pronunciations (`concordance dictcom-pron`).

A fetch stage like `wordnik-pron`: for each active word with no valid IPA,
read the word's dictionary.com page and keep the IPA it prints, as-is, in
word.dictcom_ipa (Random House, American) / word.dictcom_ipa_uk (Collins,
British). resolve_pronunciation turns those into word.ipa (tiers
DICTIONARY_COM / DICTIONARY_COM_UK) on the next `ipa` step, and the audio
upgrade then voices the word from that IPA instead of its spelling.

Fetching is plain HTTP, no bot-check involved (robots.txt allows /browse/),
at about one page a second. Only an exact headword counts: dictionary.com
redirects a word it doesn't hold to its closest spelling ("bewtifool" ->
"befool"), and a redirect is a miss, never followed. If the site starts
refusing (403/429 that won't clear), the run stops and leaves the rest
unchecked rather than pushing on.

Measured 2026-11-10 on 100 random spelling-guess words: ~10% have an entry
with IPA. The rest are dialect/archaic spellings, nonce words and inflections
that no general dictionary holds.
"""

from __future__ import annotations

import html as _html
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import quote

import requests

from .db import _safe_schema

BASE_URL = "https://www.dictionary.com/browse/"
USER_AGENT = "Mozilla/5.0 (compatible; Concordance vocabulary tool; https://vocab.brfinnegan.org)"
DELAY = 1.0
# consecutive refusals (403/429/5xx after retries) before the run stops
MAX_REFUSALS = 5

_BOX = 'class="box-headword-audio-pronunciation"'
_HEADWORD = re.compile(r"<(h1|span) class='hdr-headword'[^>]*>(.*?)</\1>", re.S)
_GROUP = re.compile(r'txt-entry-group-type (\w+)')
_IPA = re.compile(r'class="txt-ipa">([^<]*)<')
_POS = re.compile(r'class="box-pos-label[^"]*"[^>]*>\s*<h2>(.*?)</h2>', re.S)
_TAGS = re.compile(r"<[^>]+>")


@dataclass
class Entry:
    headword: str
    group: str          # 'dcom' (American) | 'british' | others (science, idiom...)
    ipa: str | None
    pos: list[str] = field(default_factory=list)


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(_TAGS.sub(" ", fragment))).strip()


def parse_entries(page: str) -> list[Entry]:
    """Every headword block on a page, in page order."""
    entries = []
    for chunk in page.split(_BOX)[1:]:
        head = _HEADWORD.search(chunk)
        group = _GROUP.search(chunk)
        if not head or not group:
            continue
        # (a homograph's number is a <sup> after the headword element)
        headword = _text(head.group(2))
        ipa = _IPA.search(chunk)
        entries.append(Entry(headword=headword, group=group.group(1).lower(),
                             ipa=_html.unescape(ipa.group(1)) if ipa else None,
                             pos=[_text(p).lower() for p in _POS.findall(chunk)]))
    return entries


def clean_ipa(raw: str | None) -> str | None:
    """'/ hjuˈmɛk tənt, ju- /' -> 'hjuˈmɛktənt': the first full variant,
    without the slashes and the spaces dictionary.com puts between
    syllables. None when the first variant is itself partial ("-ˈnjut")."""
    if not raw:
        return None
    first = raw.strip().strip("/").split(",")[0].split(";")[0].strip()
    if not first or "-" in first or "…" in first:
        return None
    return re.sub(r"\s+", "", first) or None


def _norm(word: str) -> str:
    return re.sub(r"\s+", " ", word.replace("·", "").replace("ˈ", "")).strip().lower()


def choose_ipa(entries: list[Entry], lemma: str, pos: str | None = None) -> tuple[str | None, str | None]:
    """(american, british) IPA for this exact headword. Homographs that
    disagree (minute: ˈmɪnɪt / maɪˈnut) are settled by part of speech; with
    no match, that dialect gives nothing rather than a guess."""
    want = _norm(lemma)
    pos = (pos or "").strip().lower()
    out = []
    for group in ("dcom", "british"):
        cands = [(clean_ipa(e.ipa), e.pos) for e in entries
                 if e.group == group and _norm(e.headword) == want]
        cands = [(ipa, p) for ipa, p in cands if ipa]
        distinct = {ipa for ipa, _ in cands}
        if len(distinct) == 1:
            out.append(distinct.pop())
        elif distinct and pos:
            # an entry led by this part of speech beats one that only lists it
            match = (next((ipa for ipa, p in cands if p and p[0].startswith(pos)), None)
                     or next((ipa for ipa, p in cands if any(label.startswith(pos) for label in p)), None))
            out.append(match)
        else:
            out.append(None)
    return out[0], out[1]


def fetch_page(session: requests.Session, lemma: str, tries: int = 3) -> tuple[str, str | None]:
    """('ok', html) | ('miss', None) for no exact entry | ('refused', None)."""
    url = BASE_URL + quote(lemma.strip().replace(" ", "-"), safe="")
    wait = 5.0
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=20, allow_redirects=False)
        except requests.RequestException:
            r = None
        if r is not None:
            if r.status_code == 200:
                return "ok", r.text
            if r.status_code in (301, 302, 303, 307, 308, 404, 410):
                return "miss", None
        if attempt < tries - 1:
            time.sleep(wait)
            wait *= 3
    return "refused", None


def candidates(conn, schema: str) -> list[tuple[int, str, str | None]]:
    """Unchecked active words with no IPA (or only a derived one),
    spelling-guess audio first:
    (id, lemma, part of speech)."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.lemma, w.part_of_speech
                        FROM {s}.word w LEFT JOIN {s}.word_audio a ON a.word_id = w.id
                        WHERE w.active AND w.dictcom_checked_at IS NULL
                          AND (coalesce(w.ipa, '') = '' OR w.ipa_source = 'derived')
                        ORDER BY CASE a.source WHEN 'piper' THEN 0 WHEN 'azure_guess' THEN 1
                                               WHEN 'none' THEN 2 ELSE 3 END, w.id""")
        return cur.fetchall()


def fetch_pronunciations(conn, schema: str, *, limit: int = 0, delay: float = DELAY, log=print) -> dict:
    """One run over the candidates; resumable (dictcom_checked_at). A word
    that gains IPA here gets its ipa_checked_at cleared, so the next `ipa`
    step resolves it again with this tier available."""
    s = _safe_schema(schema)
    rows = candidates(conn, schema)
    if limit:
        rows = rows[:limit]
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    stats: Counter = Counter(candidates=len(rows))
    refusals = 0
    with conn.cursor() as cur:
        for wid, lemma, pos in rows:
            outcome, page = fetch_page(session, lemma)
            if outcome == "refused":
                refusals += 1
                stats["refused"] += 1
                if refusals >= MAX_REFUSALS:
                    stats["stopped_refused"] = 1
                    log(f"  dictionary.com refused {refusals} in a row -- stopping; the rest stay unchecked")
                    break
                time.sleep(delay)
                continue
            refusals = 0
            us = uk = None
            if page is not None:
                us, uk = choose_ipa(parse_entries(page), lemma, pos)
            if us or uk:
                stats["found_us" if us else "found_uk_only"] += 1
                cur.execute(f"""UPDATE {s}.word SET dictcom_ipa = %s, dictcom_ipa_uk = %s,
                                    dictcom_checked_at = now(), ipa_checked_at = NULL WHERE id = %s""",
                            (us, uk, wid))
            else:
                stats["no_entry" if page is None else "no_ipa"] += 1
                cur.execute(f"UPDATE {s}.word SET dictcom_checked_at = now() WHERE id = %s", (wid,))
            stats["checked"] += 1
            if stats["checked"] % 50 == 0:
                conn.commit()
                log(f"  {stats['checked']}/{len(rows)} checked, "
                    f"{stats['found_us'] + stats['found_uk_only']} with IPA")
            time.sleep(delay)
    conn.commit()
    return dict(stats)
