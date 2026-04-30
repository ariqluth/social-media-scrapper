"""Text preprocessing for Indonesian + English social-media captions.

Pipeline:
    raw → lowercase → URL/mention/hashtag strip → emoji removal →
    punctuation normalise → (optional) stopword removal → (optional) stemming.
"""

from __future__ import annotations

import re
import string
from typing import Iterable, List, Optional


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"    # symbols & pictographs
    "\U0001F680-\U0001F6FF"    # transport & map
    "\U0001F1E0-\U0001F1FF"    # flags
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_MULTISPACE_RE = re.compile(r"\s+")


# Minimal stopword lists. Install nltk / sastrawi for full coverage.
_STOPWORDS_ID = {
    "yang", "di", "ke", "dari", "untuk", "dan", "atau", "ini", "itu",
    "dengan", "pada", "adalah", "tidak", "akan", "sudah", "saya", "kita",
    "kami", "mereka", "juga", "agar", "seperti", "tapi", "tetapi", "karena",
}
_STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "but", "if", "while", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "to", "of", "in", "on", "at", "by", "for", "with",
    "from", "as", "this", "that", "these", "those", "it", "its",
}


class TextCleaner:
    """Reusable cleaner. Safe for both Indonesian and English.

    Heavy options (stemming, stopword removal via Sastrawi / NLTK) are
    opt-in so the module can be imported without those heavy deps.
    """

    def __init__(
        self,
        lower: bool = True,
        strip_urls: bool = True,
        strip_mentions: bool = True,
        strip_hashtags: bool = False,      # keep hashtag tokens by default
        strip_emoji: bool = True,
        strip_punct: bool = True,
        remove_stopwords: bool = False,
        language: str = "id",              # "id" | "en" | "mixed"
        stem: bool = False,
    ):
        self.lower = lower
        self.strip_urls = strip_urls
        self.strip_mentions = strip_mentions
        self.strip_hashtags = strip_hashtags
        self.strip_emoji = strip_emoji
        self.strip_punct = strip_punct
        self.remove_stopwords = remove_stopwords
        self.language = language
        self.stem = stem

        self._stemmer = None
        if stem and language in ("id", "mixed"):
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # type: ignore
                self._stemmer = StemmerFactory().create_stemmer()
            except ImportError:
                self._stemmer = None  # silently skip if Sastrawi missing

    # ------------------------------------------------------------------

    def clean(self, text: Optional[str]) -> str:
        if text is None:
            return ""
        s = str(text)
        if self.lower:
            s = s.lower()
        if self.strip_urls:
            s = _URL_RE.sub(" ", s)
        if self.strip_mentions:
            s = _MENTION_RE.sub(" ", s)
        if self.strip_hashtags:
            s = _HASHTAG_RE.sub(" ", s)
        else:
            # keep the word but drop the leading '#'
            s = _HASHTAG_RE.sub(lambda m: m.group(0)[1:], s)
        if self.strip_emoji:
            s = _EMOJI_RE.sub(" ", s)
        if self.strip_punct:
            s = s.translate(str.maketrans("", "", string.punctuation))
        s = _MULTISPACE_RE.sub(" ", s).strip()

        if self.remove_stopwords:
            tokens = s.split()
            sw: set[str] = set()
            if self.language in ("id", "mixed"):
                sw |= _STOPWORDS_ID
            if self.language in ("en", "mixed"):
                sw |= _STOPWORDS_EN
            tokens = [t for t in tokens if t not in sw]
            s = " ".join(tokens)

        if self.stem and self._stemmer is not None:
            s = self._stemmer.stem(s)

        return s

    def clean_many(self, texts: Iterable[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    def __call__(self, text: Optional[str]) -> str:
        return self.clean(text)
