"""Word-pronunciation audio (§ audio pronunciation).

Sources, in priority order, per word:

  1. Real human recordings from Wikimedia Commons or Merriam-Webster — the
     best possible answer where they exist: an actual person, not a
     synthesizer.
  2. Azure Neural TTS, given the word's IPA directly via SSML's
     `<phoneme alphabet="ipa">` — a synthesized voice, but anchored to a
     verified transcription rather than guessing pronunciation from
     spelling. Kept for this role specifically (rather than a local model)
     because Azure's SSML parser errors loudly on any phoneme it doesn't
     recognize rather than silently mispronouncing: a prior local (StyleTTS2)
     alternative had comparable-or-better voice quality but silently mangled
     an unrecognized affricate glyph in testing instead of erroring — a local
     engine fed custom IPA carries that same risk, so this tier stays on
     Azure.
  3. Piper (local, grapheme-only): for words with no curated IPA anywhere —
     most of this corpus, see 0-Dict/Wordnik/kaikki/local-Wiktionary
     coverage. Piper's own espeak-ng-backed G2P guesses from spelling alone,
     same as any TTS would, but using its own well-tested text-to-phoneme
     path (no custom phoneme injection, so the StyleTTS2 failure mode above
     doesn't apply) — free, local, no rate limit, closes the gap Azure's
     IPA-guided tier structurally can't. Always recorded as a distinct,
     lower-confidence source; never conflated with IPA-guided output.

Measure the real IPA-coverage split live via word_audio.source counts, not a
hardcoded number here — it shifts every time a new book is ingested.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import requests

from .deepdef import _load_dotenv

AUDIO_DIR = Path("audio")
AZURE_ENDPOINT = "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
AZURE_VOICE = "en-US-AvaNeural"
# Verified live against the real Azure Speech resource (eastus region) before
# adopting -- a wrong/unavailable voice name fails synthesis silently
# per-word rather than loudly, so this isn't a guess.
AZURE_VOICE_UK = "en-GB-SoniaNeural"
_RETRY_STATUS = {429, 500, 502, 503, 504}


def ipa_dialect_for_source(source: str | None) -> str:
    """word.ipa_source -> 'us'/'uk' -- every source except 'oed' is US-biased
    by design (wiktextract.best_ipa prefers US-tagged entries, local
    Wiktionary's column is literally us_pronunciation, Wordnik's ARPAbet/AHD-5
    converters are both US phoneme systems). None (legacy rows predating
    ipa_source) is treated as 'us' too -- matches current behavior exactly,
    zero change for every word already filled before this existed."""
    return "uk" if source == "oed" else "us"


def voice_for_dialect(dialect: str) -> tuple[str, str]:
    """dialect -> (azure_voice, ssml_lang)."""
    if dialect == "uk":
        return AZURE_VOICE_UK, "en-GB"
    return AZURE_VOICE, "en-US"

# Combining double inverted breve (IPA tie bar, e.g. t͡ʃ). Both Azure and the
# local StyleTTS2 test were trained on/expect the decomposed two-letter form
# (tʃ), not the tied or ligature form — verified empirically (a ligature
# substitution silently produced garbled audio in testing).
_TIE = "͡"


def normalize_ipa(ipa: str, *, keep_optional: bool = True) -> str:
    """Curated IPA (Wiktionary/kaikki, possibly slash/bracket-delimited, possibly
    tie-barred) -> a plain phoneme string safe to hand to a synthesizer.

    Found via a real failure: kaikki marks an optional/dialectal sound in
    parentheses (e.g. "gibber" -> /ˈdʒɪbə(ɹ)/, the r-coloring some dialects
    drop). Literal "(" ")" aren't valid IPA/SSML phoneme characters — Azure
    silently rejected these, and 165 words with perfectly good IPA fell through
    to the no-data bucket as a result.

    keep_optional=True (default): keep the optional sound rather than drop it
    (the fuller pronunciation), just remove the parentheses themselves -- the
    right call for a US voice, where post-vocalic r is always pronounced.
    keep_optional=False: drop the whole parenthetical span. OED's (r) marks an
    RP linking/intrusive r -- pronounced only in connected speech before a
    following vowel, dropped in citation form -- so for a UK voice, KEEPING
    the letter (the True behavior) produces a rhotic mispronunciation exactly
    backwards from what the notation means. The trailing `?` in the pattern
    defensively handles a still-unclosed paren even though callers are
    expected to have already balanced it before this point."""
    ipa = ipa.strip().strip("/[]")
    ipa = ipa.replace(_TIE, "")
    ipa = ipa.replace(".", "")
    if keep_optional:
        ipa = ipa.replace("(", "").replace(")", "")
    else:
        ipa = re.sub(r"\([^)]*\)?", "", ipa)
    return ipa


# Symbols essentially never used in English IPA transcription but common in
# French/German/etc. — a page's pronunciation section occasionally cross-links a
# foreign-language cognate, and a naive scrape can grab that instead. Caught
# empirically: the pre-existing word.ipa scrape had "murmurer" -> French
# /myʁ.my.ʁe/ and "angelus" -> French/Latin /ɑ̃.ʒe.lys/, both of which would
# synthesize as badly mispronounced English otherwise.
_NON_ENGLISH_IPA = re.compile("[ʁɲɥyøœ̃]")  # last is the nasal-vowel tilde


def looks_like_english_ipa(ipa: str) -> bool:
    return bool(ipa) and not _NON_ENGLISH_IPA.search(ipa)


# --- Azure credentials -----------------------------------------------------

def azure_credentials() -> tuple[str, str] | tuple[None, None]:
    if "AZURE_SPEECH_KEY" not in os.environ:
        _load_dotenv(Path(".env"))
    key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
    region = os.environ.get("AZURE_SPEECH_REGION", "").strip()
    return (key, region) if key and region else (None, None)


# --- Tier 2: Azure IPA-guided synthesis -------------------------------------

def _synthesize_ssml(ssml: str, key: str, region: str, tries: int = 4,
                      word: str = "") -> bytes | None:
    """Returns mp3 bytes, or None with a printed reason on every path that
    gives up -- a prior version returned None bare on any non-200 or
    exhausted-retries network error, which meant an entire Azure Speech
    resource going bad (wrong/expired key, exhausted quota, region outage)
    was indistinguishable from ordinary per-word misses: a whole batch
    could silently produce zero Azure audio with nothing in the logs to
    explain why (confirmed live -- see logs/audio_20260904_123145.log,
    10,690 words processed, 0 Azure-synthesized, no error anywhere)."""
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-96kbitrate-mono-mp3",
        "User-Agent": "concordance-audio",
    }
    url = AZURE_ENDPOINT.format(region=region)
    label = f" for {word!r}" if word else ""
    delay = 0.5
    for attempt in range(tries):
        try:
            r = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=20)
        except requests.RequestException as exc:
            if attempt == tries - 1:
                print(f"  [azure] network error{label} after {tries} tries: {exc}")
                return None
            time.sleep(delay); delay *= 2; continue
        if r.status_code in _RETRY_STATUS and attempt < tries - 1:
            time.sleep(delay); delay *= 2; continue
        if r.status_code != 200:
            print(f"  [azure] HTTP {r.status_code}{label}: {r.text[:200]!r}")
            return None
        return r.content
    return None


def synthesize_azure(word: str, ipa: str, key: str, region: str,
                      voice: str = AZURE_VOICE, lang: str = "en-US",
                      keep_optional: bool = True, tries: int = 4) -> bytes | None:
    """IPA-guided: returns mp3 bytes, or None on a hard failure (bad phoneme, network).
    lang/keep_optional should match voice's dialect -- see voice_for_dialect
    and normalize_ipa's own docstring for why keep_optional flips per dialect."""
    ph = normalize_ipa(ipa, keep_optional=keep_optional)
    ssml = (
        f"<speak version='1.0' xml:lang='{lang}'>"
        f"<voice xml:lang='{lang}' name='{voice}'>"
        f"<phoneme alphabet='ipa' ph='{ph}'>{word}</phoneme>"
        "</voice></speak>"
    )
    return _synthesize_ssml(ssml, key, region, tries, word=word)


def synthesize_azure_guess(word: str, key: str, region: str,
                           voice: str = AZURE_VOICE, lang: str = "en-US",
                           tries: int = 4) -> bytes | None:
    """No IPA available anywhere for this word — Azure's own text-to-speech
    front-end guesses pronunciation from spelling alone, same as any other
    engine would. Callers MUST record this as a distinct, lower-confidence
    source (never conflate with IPA-guided output) since it's unverified."""
    ssml = (
        f"<speak version='1.0' xml:lang='{lang}'>"
        f"<voice xml:lang='{lang}' name='{voice}'>{word}</voice></speak>"
    )
    return _synthesize_ssml(ssml, key, region, tries, word=word)


# --- Tier: local grapheme-only synthesis (Piper) -----------------------------
# Optional infra, same fallback-to-None pattern as oed.pronunciation's vision
# model: missing model files degrade gracefully (falls through to 'none')
# rather than crash a run.
PIPER_MODEL_PATH = Path("models/piper/en_US-lessac-high.onnx")
PIPER_VOICE_NAME = "piper:en_US-lessac-high"

_piper_voice_cache = None


def _piper_voice():
    """Lazily loads the local Piper voice once per process (loading costs
    real time; must not happen per word)."""
    global _piper_voice_cache
    if _piper_voice_cache is None:
        try:
            from piper import PiperVoice
            _piper_voice_cache = PiperVoice.load(str(PIPER_MODEL_PATH)) or False
        except Exception:
            _piper_voice_cache = False
    return _piper_voice_cache or None


def synthesize_piper(word: str) -> bytes | None:
    """Grapheme-only synthesis for words with no verified pronunciation
    anywhere: Piper's own espeak-ng-backed G2P guesses from spelling alone,
    same as any TTS would -- this is what actually closes the gap Azure's
    IPA-guided tier can't (most of this corpus has no curated IPA at all,
    see this module's docstring). Deliberately NOT fed curated IPA as a
    phoneme override: a prior local (StyleTTS2) test found that path
    silently mangles a phoneme outside the model's training distribution
    rather than erroring loudly the way Azure's SSML parser does -- Azure
    keeps the IPA-guided role for that reason. Callers MUST record this as a
    distinct, lower-confidence source (never conflate with IPA-guided
    output) since it's unverified, same discipline as synthesize_azure_guess."""
    voice = _piper_voice()
    if voice is None:
        return None
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(word, wav_file)

    tmp_src = AUDIO_DIR / f".piper_tmp_{os.getpid()}.wav"
    dest_mp3 = tmp_src.with_suffix(".mp3")
    tmp_src.write_bytes(buf.getvalue())
    try:
        # Same stdin=DEVNULL precaution as fetch_commons_audio's ffmpeg call
        # (see its comment) -- avoid a hung/corrupt-input ffmpeg blocking on
        # an inherited terminal stdin and freezing the whole batch.
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_src), "-codec:a", "libmp3lame", "-qscale:a", "4", str(dest_mp3)],
            capture_output=True, timeout=30, stdin=subprocess.DEVNULL,
        )
        if proc.returncode == 0 and dest_mp3.exists():
            return dest_mp3.read_bytes()
        return None
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        tmp_src.unlink(missing_ok=True)
        dest_mp3.unlink(missing_ok=True)


# --- Tier 1: Commons real recordings ----------------------------------------

def fetch_commons_audio(url: str, dest_mp3: Path, tries: int = 4) -> bool:
    """Download a Commons audio file (ogg or mp3) and transcode to dest_mp3
    via ffmpeg. Returns True on success. Honors the server's Retry-After header
    on 429 (fixed exponential backoff alone wasn't patient enough — a sustained
    rate-limit block observed earlier took over a minute to clear)."""
    delay = 0.5
    content = None
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "concordance-audio (personal vocab tool)"})
        except requests.RequestException:
            if attempt == tries - 1:
                return False
            time.sleep(delay); delay *= 2; continue
        if r.status_code in _RETRY_STATUS and attempt < tries - 1:
            retry_after = r.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.strip().isdigit() else delay
            time.sleep(min(max(wait, delay), 60.0))
            delay *= 2
            continue
        if r.status_code == 200:
            content = r.content
        break
    if content is None:
        return False

    suffix = ".ogg" if url.lower().endswith(".ogg") else Path(url).suffix or ".ogg"
    tmp_src = dest_mp3.with_suffix(suffix)
    tmp_src.write_bytes(content)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_src), "-codec:a", "libmp3lame", "-qscale:a", "4", str(dest_mp3)],
            capture_output=True, timeout=30,
            # stdin=DEVNULL, not inherited: a corrupt/truncated download can
            # make ffmpeg probe for more input and block on stdin. Inherited
            # from a real terminal, ffmpeg (running in its own job-control
            # process group) then gets SIGTTIN and stops — which stops the
            # WHOLE process group, including this Python process itself, not
            # just ffmpeg (confirmed live: a `concordance audio` run froze
            # repeatedly in the same way before landing on the timeout
            # below and crashing outright on an unrelated word, "serotine").
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode == 0 and dest_mp3.exists()
    except (subprocess.TimeoutExpired, OSError):
        # One bad Commons file (corrupt/truncated download, hung ffmpeg)
        # must not take down an entire batch run — every caller loops over
        # many words with no per-call try/except of its own.
        return False
    finally:
        tmp_src.unlink(missing_ok=True)
