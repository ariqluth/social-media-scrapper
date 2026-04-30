import re
from typing import List
from playwright.sync_api import TimeoutError as PWTimeout

from .base import BaseScraper, Profile, Post


class TwitterScraper(BaseScraper):

    platform = "twitter"
    BASE_URL = "https://x.com"

    def scrape_profile(self, username: str, max_posts: int = 10) -> Profile:
        username = username.lstrip("@").strip("/")
        url = f"{self.BASE_URL}/{username}"
        self.log.info("Fetching %s", url)

        profile = Profile(platform=self.platform, username=username, profile_url=url)

        try:
            self.page.goto(url, timeout=60000, wait_until="domcontentloaded")
            self.page.wait_for_timeout(3500)
        except PWTimeout:
            self.log.warning("Timeout loading %s", url)
            return profile

        profile.display_name = self._text('[data-testid="UserName"] span') or ""
        profile.bio = self._text('[data-testid="UserDescription"]') or ""
        profile.verified = bool(
            self.page.query_selector('[data-testid="UserName"] svg[aria-label*="Verified"]')
        )
        profile.profile_image_url = self._attr(
            'a[href$="/photo"] img', "src"
        ) or self._meta("og:image")

        try:
            links = self.page.query_selector_all(
                f'a[href^="/{username}/"]'
            )
            for link in links:
                href = link.get_attribute("href") or ""
                txt = (link.inner_text() or "").strip()
                if href.endswith("/verified_followers") or href.endswith("/followers"):
                    profile.followers = self._first_count(txt) or profile.followers
                elif href.endswith("/following"):
                    profile.following = self._first_count(txt) or profile.following
        except Exception:
            pass

        try:
            header = self._text('div[data-testid="primaryColumn"] h2') or ""
            m = re.search(r"([\d.,KMB]+)\s+posts?", header, flags=re.IGNORECASE)
            if m:
                profile.post_count = self.parse_count(m.group(1))
        except Exception:
            pass

        profile.posts = self._collect_tweets(username, max_posts)
        return profile

    def _text(self, selector: str) -> str:
        try:
            el = self.page.query_selector(selector)
            if el:
                return (el.inner_text() or "").strip()
        except Exception:
            pass
        return ""

    def _attr(self, selector: str, attr: str) -> str:
        try:
            el = self.page.query_selector(selector)
            if el:
                return el.get_attribute(attr) or ""
        except Exception:
            pass
        return ""

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

    def _first_count(self, text: str):
        m = re.search(r"([\d.,KMB]+)", text or "")
        return self.parse_count(m.group(1)) if m else None

    def _collect_tweets(self, username: str, target: int) -> List[Post]:
        seen_ids = set()
        posts: List[Post] = []
        stale = 0

        for _ in range(25):
            articles = self.page.query_selector_all('article[data-testid="tweet"]')
            for art in articles:
                try:
                    link = art.query_selector('a[href*="/status/"]')
                    href = link.get_attribute("href") if link else ""
                    m = re.search(r"/([^/]+)/status/(\d+)", href or "")
                    if not m:
                        continue
                    author, tid = m.group(1), m.group(2)
                    # skip retweets/pinned from other accounts
                    if author.lower() != username.lower():
                        continue
                    if tid in seen_ids:
                        continue
                    seen_ids.add(tid)

                    text_el = art.query_selector('[data-testid="tweetText"]')
                    caption = (text_el.inner_text() if text_el else "").strip()

                    time_el = art.query_selector("time")
                    ts = time_el.get_attribute("datetime") if time_el else ""

                    reply = self._count_from(art, '[data-testid="reply"]')
                    retweet = self._count_from(art, '[data-testid="retweet"]')
                    like = self._count_from(art, '[data-testid="like"]')
                    view = self._count_from(art, 'a[href$="/analytics"]')

                    thumb_el = art.query_selector('[data-testid="tweetPhoto"] img')
                    thumb = thumb_el.get_attribute("src") if thumb_el else ""

                    posts.append(Post(
                        post_id=tid,
                        url=f"https://x.com/{author}/status/{tid}",
                        media_type="tweet",
                        caption=caption,
                        likes=like,
                        comments=reply,
                        views=view,
                        timestamp=ts or "",
                        thumbnail_url=thumb or "",
                    ))
                    if len(posts) >= target:
                        return posts
                except Exception:
                    continue

            before = len(posts)
            self.page.mouse.wheel(0, 3000)
            self.page.wait_for_timeout(1500)
            if len(posts) == before:
                stale += 1
                if stale >= 3:
                    break
            else:
                stale = 0
        return posts

    def _count_from(self, article, selector: str):
        try:
            el = article.query_selector(selector)
            if not el:
                return None
            aria = el.get_attribute("aria-label") or ""
            m = re.search(r"([\d.,KMB]+)", aria)
            if m:
                return self.parse_count(m.group(1))
            txt = (el.inner_text() or "").strip()
            m = re.search(r"([\d.,KMB]+)", txt)
            return self.parse_count(m.group(1)) if m else None
        except Exception:
            return None
