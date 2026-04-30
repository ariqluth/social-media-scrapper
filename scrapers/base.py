import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from playwright.sync_api import Page


@dataclass
class Post:
    post_id: str = ""
    url: str = ""
    media_type: str = ""          
    caption: str = ""
    likes: Optional[int] = None
    comments: Optional[int] = None
    views: Optional[int] = None
    timestamp: str = ""           
    thumbnail_url: str = ""


@dataclass
class Profile:
    platform: str = ""
    username: str = ""
    display_name: str = ""
    bio: str = ""
    followers: Optional[int] = None
    following: Optional[int] = None
    post_count: Optional[int] = None
    profile_image_url: str = ""
    profile_url: str = ""
    verified: bool = False
    posts: List[Post] = field(default_factory=list)

    def flatten(self) -> List[dict]:
        base = {
            "platform": self.platform,
            "username": self.username,
            "display_name": self.display_name,
            "bio": self.bio,
            "followers": self.followers,
            "following": self.following,
            "post_count": self.post_count,
            "profile_image_url": self.profile_image_url,
            "profile_url": self.profile_url,
            "verified": self.verified,
        }
        if not self.posts:
            empty = {
                "post_id": "",
                "post_url": "",
                "media_type": "",
                "caption": "",
                "likes": None,
                "comments": None,
                "views": None,
                "timestamp": "",
                "thumbnail_url": "",
            }
            return [{**base, **empty}]

        rows = []
        for p in self.posts:
            rows.append({
                **base,
                "post_id": p.post_id,
                "post_url": p.url,
                "media_type": p.media_type,
                "caption": p.caption,
                "likes": p.likes,
                "comments": p.comments,
                "views": p.views,
                "timestamp": p.timestamp,
                "thumbnail_url": p.thumbnail_url,
            })
        return rows


class BaseScraper:

    platform: str = ""

    def __init__(self, page: Page, logger: Optional[logging.Logger] = None):
        self.page = page
        self.log = logger or logging.getLogger(self.platform or "scraper")

    @staticmethod
    def parse_count(text: str) -> Optional[int]:
        if not text:
            return None
        s = text.strip().lower().replace(",", "").replace("\u00a0", "")
        s = s.split()[0] if s else s
        if not s:
            return None
        mult = 1
        if s.endswith("k"):
            mult, s = 1_000, s[:-1]
        elif s.endswith("m"):
            mult, s = 1_000_000, s[:-1]
        elif s.endswith("b"):
            mult, s = 1_000_000_000, s[:-1]
        try:
            return int(float(s) * mult)
        except ValueError:
            return None

    def scrape_profile(self, username: str, max_posts: int = 10) -> Profile:
        raise NotImplementedError
