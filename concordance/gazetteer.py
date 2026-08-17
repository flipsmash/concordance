"""Curated names/places gazetteer -- a DISQUALIFYING signal for validity.py's
proper-noun check, not a vouch (see DESIGN.md's "still-mostly-unclosed gap"
section for why SymSpell/WordNet/wordfreq can't do this job: all three are
frequency-derived from general web text, so a real name with any web
footprint looks "attested" to every one of them).

Three sources:
  - given names: NLTK's `names` corpus (~7.9k, US-census-derived) -- already
    a dependency (tokenize.py/validity.py both use adjacent NLTK corpora),
    no separate download.
  - surnames: US Census 2010 surname file, every surname with >=100
    occurrences (~162k) -- data/census_surnames/Names_2010Census.csv.
  - places: GeoNames `cities1000` dump, every populated place with
    population >=1000 (~140k) -- data/geonames/cities1000.txt.

Neither file ships in the repo (data/ is gitignored, same as the kaikki
Wiktextract dump) -- download them yourself:

    curl -o data/census_surnames.zip \
      https://www2.census.gov/topics/genealogy/2010surnames/names.zip
    curl -o data/geonames_cities1000.zip \
      https://download.geonames.org/export/dump/cities1000.zip

then unzip into data/census_surnames/ and data/geonames/ respectively
(`Names_2010Census.csv` and `cities1000.txt` are the files actually read;
each zip also carries other files not needed here).
"""

from __future__ import annotations

import csv
from pathlib import Path

DEFAULT_CENSUS_PATH = Path("data/census_surnames/Names_2010Census.csv")
DEFAULT_GEONAMES_PATH = Path("data/geonames/cities1000.txt")


def load_surnames(path: Path | str = DEFAULT_CENSUS_PATH) -> set[str]:
    """US Census 2010 surnames, >=100 occurrences (the file's own inclusion
    floor -- not re-filtered here). Excludes the file's own "ALL OTHER
    NAMES" aggregate row (rank=0, a summary of everything below the floor,
    not a real surname)."""
    path = Path(path)
    names: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("name") or "").strip().lower()
            if name and row.get("rank", "0") != "0" and " " not in name:
                names.add(name)
    return names


def load_places(path: Path | str = DEFAULT_GEONAMES_PATH) -> set[str]:
    """GeoNames cities1000 dump -- populated places with population >=1000
    (the file's own inclusion floor, not re-filtered here). Uses asciiname
    (column 3 of the tab-separated dump), not the diacritic-bearing name
    column, to match this project's lowercase-ASCII lemma convention
    throughout (extraction only ever hands the validity gate tok.is_alpha
    tokens). Multi-word place names ("Sant Julia de Loria") are skipped --
    the pipeline tokenizes word by word, so a multi-word gazetteer entry
    could never match a single lemma anyway."""
    path = Path(path)
    names: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            name = cols[2].strip().lower()
            if name and " " not in name:
                names.add(name)
    return names


def load_given_names() -> set[str]:
    """NLTK's `names` corpus -- census-derived first names, both genders.
    Lazy-downloads the corpus data (not the gazetteer's own two files) on
    first use, same graceful-fetch pattern validity.py/tokenize.py already
    use for wordnet/words."""
    try:
        from nltk.corpus import names
        names.words()
    except LookupError:
        import nltk
        nltk.download("names", quiet=True)
        from nltk.corpus import names
    return {n.strip().lower() for n in names.words()}
