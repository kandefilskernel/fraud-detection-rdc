"""
Base de connaissances de l'assistant d'enquête :
    - archive des cas passés (vecteurs + description), voir ml/rag/build_case_archive.py ;
    - procédures internes (Markdown), découpées par section « ## » et indexées en BM25.

Un en-tête par fichier indique les typologies et les règles concernées : une section est
favorisée quand elle correspond à l'hypothèse de typologie ou à une règle déclenchée.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import settings
from shared.investigation.case_index import CaseIndex

DEFAULT_DIR = Path(__file__).absolute().parents[2] / "knowledge"
STOPWORDS = set("""a au aux avec ce ces cet cette d dans de des du elle en est et il ils la le les leur lui
ne ou par pas plus pour qu que qui s sa se ses son sur un une y l n c j m t si sans ne mais
donc est sont ete etre avoir a""".split())


def tokenize(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return [w for w in re.findall(r"[a-z0-9_]+", t) if w not in STOPWORDS and len(w) > 1]


@dataclass
class Procedure:
    ref: str                 # nom de fichier#section, stable (journal d'audit)
    title: str               # titre du document
    section: str
    text: str
    typologies: set[str] = field(default_factory=set)
    rules: set[str] = field(default_factory=set)


def _front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    head, _, body = raw[3:].partition("\n---")
    meta = {}
    for line in head.strip().splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    return meta, body


def load_procedures(directory: Path) -> list[Procedure]:
    out = []
    for path in sorted(directory.glob("*.md")):
        meta, body = _front_matter(path.read_text(encoding="utf-8"))
        split = lambda v: {s.strip() for s in v.split(",") if s.strip()}  # noqa: E731
        typos, rules = split(meta.get("typologies", "")), split(meta.get("regles", ""))
        for block in re.split(r"^## ", body, flags=re.M)[1:]:
            section, _, text = block.partition("\n")
            out.append(Procedure(f"{path.stem}#{tokenize(section)[0] if tokenize(section) else 'section'}",
                                 meta.get("titre", path.stem), section.strip(), text.strip(), typos, rules))
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.docs, self.k1, self.b = [Counter(d) for d in docs], k1, b
        self.lengths = [len(d) for d in docs]
        self.avg = sum(self.lengths) / max(len(docs), 1)
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, length in zip(self.docs, self.lengths):
            s = 0.0
            for w in set(query):
                tf = d.get(w, 0)
                if tf:
                    s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


class Knowledge:
    def __init__(self, directory: Path):
        self.directory = directory
        self.cases = CaseIndex.load(directory)
        self.procedures = load_procedures(directory / "procedures")
        self.bm25 = BM25([tokenize(f"{p.title} {p.section} {p.text}") for p in self.procedures])

    def search_procedures(self, query: str, typologies: list[str], rules: list[str], k: int = 4) -> list[Procedure]:
        """BM25 sur le texte + bonus si la section vise la typologie supposée ou une règle
        déclenchée. Le cadre général (typologie TOUTES) n'est retenu que s'il est pertinent."""
        base = self.bm25.scores(tokenize(query))
        top = max(base) if base and max(base) > 0 else 1.0
        scored = []
        for p, s in zip(self.procedures, base):
            bonus = 0.0
            if p.typologies & set(typologies[:1]):
                bonus += 0.6
            elif p.typologies & set(typologies):
                bonus += 0.3
            if p.rules & set(rules):
                bonus += 0.5
            scored.append((s / top + bonus, p))
        scored.sort(key=lambda x: -x[0])
        return [p for s, p in scored[:k] if s > 0]


@lru_cache(maxsize=1)
def get_knowledge() -> Knowledge:
    return Knowledge(Path(settings.KNOWLEDGE_DIR) if settings.KNOWLEDGE_DIR else DEFAULT_DIR)
