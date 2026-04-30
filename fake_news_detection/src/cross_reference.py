
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import numpy as np


VERIFIED_SOURCES: Dict[str, Dict[str, str]] = {
    "komdigi": {
        "name": "Kementerian Komunikasi dan Digital RI",
        "url": "https://www.komdigi.go.id/berita/berita-kementerian",
        "category": "official",
    },
    "tribratanews": {
        "name": "Tribrata News Polda Jabar",
        "url": "https://tribratanews.jabar.polri.go.id/category/hoax-buster/",
        "category": "official",
    },
    "cnbc_id": {
        "name": "CNBC Indonesia",
        "url": "https://www.cnbcindonesia.com/news",
        "category": "news",
    },
    "kompas": {
        "name": "Kompas.com",
        "url": "https://www.kompas.com/",
        "category": "news",
    },
    "cnn_id": {
        "name": "CNN Indonesia",
        "url": "https://www.cnnindonesia.com/nasional",
        "category": "news",
    },
    "tempo": {
        "name": "Tempo.co",
        "url": "https://www.tempo.co/",
        "category": "news",
    },
}


@dataclass
class Article:
    source_key: str
    source_name: str
    title: str
    url: str
    summary: str = ""
    published: str = ""


@dataclass
class Match:
    article: Article
    similarity: float


class CredibleSourceMatcher:
    """Builds a corpus from the verified sources and matches a query against it.

    Usage
    -----
    >>> m = CredibleSourceMatcher()
    >>> m.build_corpus()                 # scrape all 6 sources
    >>> matches = m.match("teks hoaks …", top_k=5)
    >>> for hit in matches:
    ...     print(hit.similarity, hit.article.source_name, hit.article.title)
    """

    def __init__(
        self,
        sources: Optional[Dict[str, Dict[str, str]]] = None,
        max_articles_per_source: int = 40,
        use_embeddings: bool = False,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        timeout: int = 15,
        user_agent: str = (
            "Mozilla/5.0 (compatible; FakeNewsResearchBot/1.0; "
            "+https://example.org/research)"
        ),
    ):
        self.sources = sources or VERIFIED_SOURCES
        self.max_articles_per_source = max_articles_per_source
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.user_agent = user_agent

        self.articles: List[Article] = []
        self._tfidf = None
        self._tfidf_matrix = None
        self._emb_model = None
        self._emb_matrix = None
        self.log = logging.getLogger("cross_reference")

    # --- public API -------------------------------------------------------

    def build_corpus(self) -> int:
        """Scrape all verified sources into `self.articles`. Returns count."""
        self.articles = []
        for key, meta in self.sources.items():
            try:
                articles = self._scrape_source(key, meta)
                self.log.info("%s: %d articles", meta["name"], len(articles))
                self.articles.extend(articles)
            except Exception as exc:  # keep going on per-source failures
                self.log.warning("Failed %s: %s", key, exc)
        self._index()
        return len(self.articles)

    def match(self, query: str, top_k: int = 5) -> List[Match]:
        """Return the `top_k` most similar verified articles for `query`."""
        if not self.articles:
            raise RuntimeError("Corpus is empty — call build_corpus() first.")
        if self.use_embeddings and self._emb_matrix is not None:
            sims = self._embed_sim(query)
        else:
            sims = self._tfidf_sim(query)
        idx = np.argsort(sims)[::-1][:top_k]
        return [Match(self.articles[i], float(sims[i])) for i in idx]

    # --- scraping ---------------------------------------------------------

    def _scrape_source(self, key: str, meta: Dict[str, str]) -> List[Article]:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": self.user_agent}
        r = requests.get(meta["url"], headers=headers, timeout=self.timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        articles: List[Article] = []

        seen: set[str] = set()

        for art in soup.select("article")[: self.max_articles_per_source * 2]:
            a = art.find("a", href=True)
            if not a:
                continue
            title = (a.get_text(strip=True) or "").strip()
            if len(title) < 12:
                continue
            href = urljoin(meta["url"], a["href"])
            if href in seen:
                continue
            seen.add(href)
            summary_el = art.find("p")
            summary = summary_el.get_text(strip=True) if summary_el else ""
            articles.append(Article(
                source_key=key, source_name=meta["name"],
                title=title, url=href, summary=summary,
            ))
            if len(articles) >= self.max_articles_per_source:
                break

        if not articles:
            for a in soup.find_all("a", href=True):
                title = (a.get_text(strip=True) or "").strip()
                if len(title) < 20 or len(title) > 200:
                    continue
                href = urljoin(meta["url"], a["href"])
                if href in seen:
                    continue
                seen.add(href)
                articles.append(Article(
                    source_key=key, source_name=meta["name"],
                    title=title, url=href, summary="",
                ))
                if len(articles) >= self.max_articles_per_source:
                    break

        return articles

    # --- indexing & similarity -------------------------------------------

    def _corpus_texts(self) -> List[str]:
        return [f"{a.title}. {a.summary}".strip() for a in self.articles]

    def _index(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = self._corpus_texts()
        if not texts:
            return
        self._tfidf = TfidfVectorizer(
            ngram_range=(1, 2), max_features=30000, sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(texts)

        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._emb_model = SentenceTransformer(self.embedding_model)
                self._emb_matrix = self._emb_model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=False,
                    normalize_embeddings=True,
                )
            except Exception as exc:
                self.log.warning("Embeddings unavailable, TF-IDF only: %s", exc)
                self.use_embeddings = False

    def _tfidf_sim(self, query: str) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        q = self._tfidf.transform([query])
        sims = cosine_similarity(q, self._tfidf_matrix).ravel()
        return sims

    def _embed_sim(self, query: str) -> np.ndarray:
        v = self._emb_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
        )
        sims = (self._emb_matrix @ v.T).ravel()
        return sims
