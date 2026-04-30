from .base import Profile, Post, BaseScraper
from .instagram import InstagramScraper
from .tiktok import TikTokScraper
from .twitter import TwitterScraper

SCRAPERS = {
    "instagram": InstagramScraper,
    "tiktok": TikTokScraper,
    "twitter": TwitterScraper,
    "x": TwitterScraper,
}

__all__ = ["Profile", "Post", "BaseScraper", "SCRAPERS"]
