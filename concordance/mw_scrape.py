"""Merriam-Webster word lookup — live-site scrape (fallback, for words the
Collegiate Dictionary API doesn't have, e.g. https://api miss but the site
still resolves the word via a different/broader source than the API's own
Collegiate data).

merriam-webster.com sits behind Cloudflare's managed bot challenge (confirmed:
a plain requests/curl GET gets a 403 with `cf-mitigated: challenge` regardless
of headers — this is active bot mitigation, not a robots.txt restriction,
which does allow /dictionary/*). A real browser clears the challenge on its
own after a few seconds; Playwright drives one.

Uses a PERSISTENT browser profile (launch_persistent_context) so the cleared
Cloudflare cookie survives across CLI runs -- the challenge doesn't need to be
solved on every lookup, only ever again if it expires or the site re-flags the
profile. Defaults to headed (not headless): headless Chromium is far more
likely to get flagged by Cloudflare, and on a brand-new profile a human may
need to solve an interactive checkbox once.

Selectors below were verified against two real snapshots of the rendered page
(Wayback Machine, Dec 2025 and Jul 2026 captures of /dictionary/concordance
and /dictionary/run) since the live site can't be fetched headlessly to
inspect. MW can change its markup at any time -- this is a fallback tier,
kept deliberately best-effort: a missing selector just means an entry is
missing that field, not a crash.

Etymology and "First Known Use" render ONCE per page, combined across all
homograph entries (verb/noun/adjective senses share one Word History block),
not per entry -- so that combined text is attached to every MWEntry returned
here rather than trying to split it back out per part of speech.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup

from .mw import _AUDIO_URL, _audio_subdir, MWEntry, MWPronunciation

_ENTRY_URL = "https://www.merriam-webster.com/dictionary/{word}"
DEFAULT_PROFILE_DIR = Path(".cache/mw_browser_profile")
_WAIT_SELECTOR = "div.entry-word-section-container, div.widget.content-section-with-header"


class MWScraper:
    """Holds one browser/context open across a batch of lookups.

    launch_persistent_context locks its user-data-dir for as long as the
    context is open -- opening a fresh one per word (the original design)
    meant back-to-back calls could race that lock and fail spuriously.
    Reusing a single context for the run's whole word list avoids that, and
    is also just faster (no relaunch per word)."""

    def __init__(self, profile_dir: Path = DEFAULT_PROFILE_DIR, headless: bool = False):
        self.profile_dir = profile_dir
        self.headless = headless
        self._pw = None
        self._context = None

    def __enter__(self) -> "MWScraper":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return self
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        self._context = self._pw.chromium.launch_persistent_context(
            str(self.profile_dir), headless=self.headless)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._context is not None:
            self._context.close()
        if self._pw is not None:
            self._pw.stop()

    def lookup(self, word: str, timeout_ms: int = 20000) -> list[MWEntry]:
        """Returns [] on a genuine miss, a navigation timeout, or if
        playwright/its browser isn't available -- callers already treat an
        empty API result as "nothing found here," so this degrades the same
        way. Prints the reason to stderr rather than swallowing it silently,
        since "no entry found" otherwise conflates a real miss with a
        Cloudflare block, a locked profile, or a missing display."""
        if self._context is None:
            print("MW scrape: playwright not installed (pip install concordance[scrape])", file=sys.stderr)
            return []
        try:
            page = self._context.new_page()
            page.goto(_ENTRY_URL.format(word=quote(word)), timeout=timeout_ms)
            page.wait_for_selector(_WAIT_SELECTOR, timeout=timeout_ms)
            html = page.content()
            page.close()
        except Exception as exc:
            print(f"MW scrape failed for {word!r}: {exc}", file=sys.stderr)
            return []
        return _parse_page(html, word)


def scrape_word(word: str, profile_dir: Path = DEFAULT_PROFILE_DIR,
                 headless: bool = False, timeout_ms: int = 20000) -> list[MWEntry]:
    """One-off convenience wrapper (single word, own browser lifecycle). For
    more than one word in a run, use MWScraper directly and reuse it."""
    with MWScraper(profile_dir=profile_dir, headless=headless) as scraper:
        return scraper.lookup(word, timeout_ms=timeout_ms)


def _parse_page(html: str, word: str) -> list[MWEntry]:
    soup = BeautifulSoup(html, "html.parser")

    ety_el = soup.find("p", class_="et")
    etymology = ety_el.get_text(" ", strip=True) if ety_el else ""
    fku_el = soup.find("p", class_="ety-sl")
    first_known_use = fku_el.get_text(" ", strip=True) if fku_el else ""

    entries = []
    for container in soup.find_all("div", class_="entry-word-section-container"):
        entry = _parse_entry_container(container, word, etymology, first_known_use)
        if entry:
            entries.append(entry)
    return entries


def _parse_entry_container(container, word: str, etymology: str, first_known_use: str) -> MWEntry | None:
    hw_el = container.find("h1", class_="hword")
    pos_el = container.find("h2", class_="parts-of-speech")
    headword = hw_el.get_text(strip=True) if hw_el else word
    pos = pos_el.get_text(strip=True) if pos_el else ""

    pronunciations = []
    for a in container.find_all("a", class_="play-pron-v2"):
        # Only the anchor's own direct text -- get_text() would also sweep in
        # the nested play-icon <svg><title>How to pronounce...</title></svg>,
        # an accessibility label, not part of the respelling.
        respelling = "".join(c for c in a.contents if isinstance(c, str)).strip()
        audio_file = a.get("data-file")
        audio_dir = a.get("data-dir") or (_audio_subdir(audio_file) if audio_file else None)
        audio_url = _AUDIO_URL.format(subdir=audio_dir, file=audio_file) if audio_file else None
        if respelling or audio_url:
            pronunciations.append(MWPronunciation(respelling=respelling, audio_url=audio_url))

    # MW prefixes each sense with a literal bold colon in its own markup (the API's
    # {bc} token renders the same way) -- strip it here so scraped and API-sourced
    # definitions read identically instead of scraped ones alone keeping "1. : ...".
    definitions = [
        dt.get_text(" ", strip=True).lstrip(": ").strip()
        for dt in container.find_all("span", class_="dtText")
    ]
    if not definitions:
        return None

    return MWEntry(
        headword=headword, part_of_speech=pos, definitions=definitions,
        pronunciations=pronunciations, etymology=etymology,
        first_known_use=first_known_use, source="Merriam-Webster (scraped)",
    )
