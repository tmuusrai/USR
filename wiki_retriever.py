"""
Phase 3 — Wiki-native BM25 retrieval.
Replaces FAISS chunk-based search with full wiki-page retrieval.
No external API needed; runs entirely offline.
"""
import re
import math
from pathlib import Path
from collections import Counter

LLM_WIKI_DIR = Path("llm_wiki_data")
CONCEPTS_DIR = LLM_WIKI_DIR / "concepts"

_CJK = re.compile(r'[一-鿿㐀-䶿]')
_EN  = re.compile(r'[A-Za-z0-9]+')


def _tokenize(text: str) -> list[str]:
    tokens = []
    cjk_chars = _CJK.findall(text)
    tokens.extend(cjk_chars)                                   # unigrams
    for i in range(len(cjk_chars) - 1):                       # bigrams
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])
    for w in _EN.findall(text):
        tokens.append(w.lower())
    return tokens


def _safe_read(path: Path) -> str | None:
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return None


class WikiRetriever:
    """BM25 over all llm_wiki_data pages (concept hubs + plan wikis)."""
    K1 = 1.5
    B  = 0.75

    def __init__(self):
        self.pages:  list[dict]    = []
        self._tf:    list[Counter] = []
        self._df:    Counter       = Counter()
        self._avgdl: float         = 300.0
        self._build()

    def _build(self):
        paths: list[tuple[str, Path]] = []
        if CONCEPTS_DIR.exists():
            for p in sorted(CONCEPTS_DIR.glob("*.md")):
                paths.append(("concept", p))
        if LLM_WIKI_DIR.exists():
            for p in sorted(LLM_WIKI_DIR.glob("*.md")):
                paths.append(("plan", p))

        total_len = 0
        for ptype, path in paths:
            text = _safe_read(path)
            if not text:
                continue
            title = path.stem
            # title repeated 3× gives it higher term weight
            tokens = _tokenize(title * 3 + " " + text)
            tf = Counter(tokens)
            total_len += len(tokens)
            self.pages.append({"title": title, "content": text, "type": ptype})
            self._tf.append(tf)
            for term in tf:
                self._df[term] += 1

        n = len(self.pages)
        self._avgdl = total_len / n if n else 300.0
        n_concept = sum(1 for p in self.pages if p["type"] == "concept")
        n_plan    = n - n_concept
        print(f"[WikiRAG] 索引完成：{n} 頁（概念 {n_concept} + 計畫 {n_plan}）")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []

        n = len(self.pages)
        scores: list[tuple[float, int]] = []

        for i, (page, tf) in enumerate(zip(self.pages, self._tf)):
            dl = sum(tf.values())
            score = 0.0
            for term in q_tokens:
                df = self._df.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_val = tf.get(term, 0)
                norm = tf_val * (self.K1 + 1) / (
                    tf_val + self.K1 * (1 - self.B + self.B * dl / self._avgdl)
                )
                score += idf * norm
            if page["type"] == "concept":
                score *= 1.4   # synthesized hub pages get a boost
            scores.append((score, i))

        scores.sort(reverse=True)
        results = []
        for score, i in scores[:top_k]:
            if score > 0:
                p = dict(self.pages[i])
                p["score"] = round(score, 3)
                results.append(p)
        return results
