"""BM25 retrieval over knowledge-base articles, split into sentence-level chunks."""
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9_]+")
STOP = {"the", "a", "an", "to", "of", "and", "or", "is", "in", "on", "for", "my", "i", "it", "can", "be", "with", "from", "after", "not", "that", "this", "their", "when", "we", "our", "me", "you", "has", "have", "are", "was", "does"}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass(frozen=True)
class Chunk:
    article: str
    title: str
    text: str


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.title + " " + c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_directory(cls, path: str | Path) -> "BM25Index":
        chunks = []
        for f in sorted(Path(path).glob("*.md")):
            lines = f.read_text().strip().splitlines()
            title = lines[0].lstrip("# ").strip()
            for line in lines[1:]:
                if line.strip():
                    chunks.append(Chunk(f.stem, title, line.strip()))
        return cls(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = []
        for chunk, doc, length in zip(self.chunks, self.docs, self.lengths):
            s = 0.0
            for t in q:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if s > 0:
                scored.append((chunk, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
