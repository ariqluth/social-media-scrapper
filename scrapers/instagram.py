import json
import re
from typing import List, Optional
from playwright.sync_api import TimeoutError as PWTimeout

from .base import BaseScraper, Profile, Post


class InstagramScraper(BaseScraper):

    platform = "instagram"
    BASE_URL = "https://www.instagram.com"
    _BIO_RE = re.compile(r'on Instagram:\s*"(.+)"\s*\Z', re.DOTALL | re.IGNORECASE)
    _COUNTS_RE = re.compile(
        r"([\d.,KMB\s]+)\s+Followers?,\s+([\d.,KMB\s]+)\s+Following,\s+([\d.,KMB\s]+)\s+Posts?",
        re.IGNORECASE,
    )
    _DISPLAY_NAME_RE = re.compile(r"(.+?)\s*\(@")
    _CAPTION_TITLE_RE = re.compile(r'on Instagram:\s*"(.+)"', re.DOTALL | re.IGNORECASE)
    _CAPTION_DESC_RE = re.compile(r':\s*"(.+)"\s*\.?\s*\Z', re.DOTALL)
    _LIKES_RE = re.compile(r"([\d.,KMB]+)\s+likes?", re.IGNORECASE)
    _COMMENTS_RE = re.compile(r"([\d.,KMB]+)\s+comments?", re.IGNORECASE)
    _VIEWS_RE = re.compile(r"([\d.,KMB]+)\s+views?", re.IGNORECASE)

    def scrape_profile(self, username: str, max_posts: int = 10) -> Profile:
        username = username.lstrip("@").strip("/")
        url = f"{self.BASE_URL}/{username}/"
        self.log.info("Fetching %s", url)

        profile = Profile(platform=self.platform, username=username, profile_url=url)

        try:
            self.page.goto(url, timeout=60000, wait_until="domcontentloaded")
            self.page.wait_for_timeout(2500)
        except PWTimeout:
            self.log.warning("Timeout loading %s", url)
            return profile

        # counts
        desc_og = self._meta("og:description")
        desc_name = self._meta("description")
        for d in (desc_name, desc_og):
            if not d:
                continue
            m = self._COUNTS_RE.search(d)
            if m:
                profile.followers = self.parse_count(m.group(1))
                profile.following = self.parse_count(m.group(2))
                profile.post_count = self.parse_count(m.group(3))
                break

        # bio (name="description" only)
        if desc_name:
            m = self._BIO_RE.search(desc_name)
            if m:
                profile.bio = m.group(1).strip()

        # display name from og:title / <title>
        title = self._meta("og:title")
        if not title:
            try:
                title = self.page.title() or ""
            except Exception:
                title = ""
        if title:
            m = self._DISPLAY_NAME_RE.match(title)
            if m:
                profile.display_name = m.group(1).strip()
        if not profile.display_name and desc_name:
            tail = desc_name.split(" - ", 1)[-1] if " - " in desc_name else ""
            m = self._DISPLAY_NAME_RE.match(tail)
            if m:
                profile.display_name = m.group(1).strip()

        profile.profile_image_url = self._meta("og:image") or ""

        # posts / reels
        shortcodes = self._collect_shortcodes(max_posts)
        self.log.info("Found %d post shortcodes", len(shortcodes))

        for sc in shortcodes[:max_posts]:
            post = self._scrape_post(sc)
            if post:
                profile.posts.append(post)

        return profile

    def _meta(self, prop: str) -> str:
        try:
            el = self.page.query_selector(
                f'meta[property="{prop}"], meta[name="{prop}"]'
            )
            if el:
                return el.get_attribute("content") or ""
        except Exception:
            pass
        return ""

    def _collect_shortcodes(self, target: int) -> List[str]:
        seen: List[str] = []
        stale = 0
        for _ in range(20):
            hrefs = self.page.eval_on_selector_all(
                'a[href*="/p/"], a[href*="/reel/"]',
                "els => els.map(e => e.getAttribute('href'))",
            )
            for h in hrefs or []:
                m = re.search(r"/(p|reel)/([^/?#]+)", h or "")
                if m:
                    sc = f"{m.group(1)}/{m.group(2)}"
                    if sc not in seen:
                        seen.append(sc)
            if len(seen) >= target:
                break
            before = len(seen)
            self.page.mouse.wheel(0, 3000)
            self.page.wait_for_timeout(1500)
            if len(seen) == before:
                stale += 1
                if stale >= 3:
                    break
            else:
                stale = 0
        return seen

    def _scrape_post(self, shortcode_path: str) -> Optional[Post]:
        kind, code = shortcode_path.split("/", 1)
        url = f"{self.BASE_URL}/{kind}/{code}/"
        media_type = "reel" if kind == "reel" else "image"

        try:
            self.page.goto(url, timeout=45000, wait_until="domcontentloaded")
            self.page.wait_for_timeout(1800)
        except PWTimeout:
            return Post(post_id=code, url=url, media_type=media_type)

        post = Post(post_id=code, url=url, media_type=media_type)
        post.thumbnail_url = self._meta("og:image") or ""

        desc = self._meta("og:description") or self._meta("description") or ""
        og_title = self._meta("og:title") or ""

        # engagement counts
        m = self._LIKES_RE.search(desc)
        if m:
            post.likes = self.parse_count(m.group(1))
        m = self._COMMENTS_RE.search(desc)
        if m:
            post.comments = self.parse_count(m.group(1))
        m = self._VIEWS_RE.search(desc)
        if m:
            post.views = self.parse_count(m.group(1))

        # caption: og:title is cleanest, fall back to og:description
        m = self._CAPTION_TITLE_RE.search(og_title)
        if not m:
            m = self._CAPTION_DESC_RE.search(desc)
        if m:
            post.caption = m.group(1).strip()

        # ld+json for timestamp / backup caption
        try:
            for script in self.page.query_selector_all('script[type="application/ld+json"]'):
                raw = script.inner_text()
                if not raw:
                    continue
                data = json.loads(raw)
                if isinstance(data, list):
                    data = data[0] if data else {}
                if isinstance(data, dict):
                    if not post.timestamp and data.get("uploadDate"):
                        post.timestamp = data["uploadDate"]
                    if not post.timestamp and data.get("datePublished"):
                        post.timestamp = data["datePublished"]
                    if not post.caption and data.get("caption"):
                        post.caption = str(data["caption"]).strip()
        except Exception:
            pass

        # final fallback for timestamp
        if not post.timestamp:
            try:
                t = self.page.query_selector("time[datetime]")
                if t:
                    post.timestamp = t.get_attribute("datetime") or ""
            except Exception:
                pass

        return post
