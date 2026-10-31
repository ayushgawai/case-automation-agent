"""Multinomial naive Bayes case classifier with a confidence score for escalation."""
import math
from collections import Counter, defaultdict

from .retrieval import tokenize


class NaiveBayesTriage:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_counts: Counter = Counter()
        self.token_counts: dict[str, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, texts: list[str], labels: list[str]) -> "NaiveBayesTriage":
        for text, label in zip(texts, labels):
            toks = tokenize(text)
            self.class_counts[label] += 1
            self.token_counts[label].update(toks)
            self.vocab.update(toks)
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        toks = tokenize(text)
        total = sum(self.class_counts.values())
        v = len(self.vocab)
        logp = {}
        for c, n in self.class_counts.items():
            denom = sum(self.token_counts[c].values()) + self.alpha * v
            logp[c] = math.log(n / total) + sum(
                math.log((self.token_counts[c][t] + self.alpha) / denom) for t in toks if t in self.vocab)
        m = max(logp.values())
        z = sum(math.exp(x - m) for x in logp.values())
        return {c: math.exp(x - m) / z for c, x in logp.items()}

    def predict(self, text: str) -> tuple[str, float]:
        p = self.predict_proba(text)
        label = max(p, key=p.get)
        return label, p[label]
