import json
import re
from typing import List
from playwright.sync_api import TimeoutError as PWTimeout

from .base import BaseScraper, Profile, Post


class TikTokScraper(BaseScraper):

    platform = "tiktok"
    BASE_URL = "https://www.tiktok.com"

    def scrape_profile(self, username: str, max_posts: int = 10) -> Profile:
        username = username.lstrip("@").strip("/")
        url = f"{self.BASE_URL}/@{username}"
        self.log.info("Fetching %s", url)

        profile = Profile(
            platform=self.platform,
            username=username,
            profile_url=url,
        )

        try:
            self.page.goto(url, timeout=60000, wait_until="domcontentloaded")
            self.page.wait_for_timeout(3000)
        except PWTimeout:
            self.log.warning("Timeout loading %s", url)
            return profile

        data = self._read_universal_data()

        user_info = {}
        stats = {}
        if data:
            try:
                scope = (
                    data.get("__DEFAULT_SCOPE__", {})
                    .get("webapp.user-detail", {})
                    .get("userInfo", {})
                )
                user_info = scope.get("user", {}) or {}
                stats = scope.get("stats", {}) or scope.get("statsV2", {}) or {}
            except Exception:
                pass

        profile.display_name = user_info.get("nickname", "") or ""
        profile.bio = user_info.get("signature", "") or ""
        profile.verified = bool(user_info.get("verified", False))
        profile.profile_image_url = (
            user_info.get("avatarLarger")
            or user_info.get("avatarMedium")
            or user_info.get("avatarThumb")
            or ""
        )
        if stats:
            profile.followers = self._as_int(stats.get("followerCount"))
            profile.following = self._as_int(stats.get("followingCount"))
            profile.post_count = self._as_int(stats.get("videoCount"))

        if not profile.display_name:
            profile.display_name = self._text('[data-e2e="user-title"]') or self._text(
                '[data-e2e="user-subtitle"]'
            )
        if not profile.bio:
            profile.bio = self._text('[data-e2e="user-bio"]') or ""
        if profile.followers is None:
            profile.followers = self.parse_count(self._text('[data-e2e="followers-count"]'))
        if profile.following is None:
            profile.following = self.parse_count(self._text('[data-e2e="following-count"]'))
        if not profile.profile_image_url:
            profile.profile_image_url = self._meta("og:image") or ""

        video_ids = self._collect_video_ids(max_posts)
        self.log.info("Found %d video ids", len(video_ids))

        for vid in video_ids[:max_posts]:
            post = self._scrape_video(username, vid)
            if post:
                profile.posts.append(post)

        return profile

    def _read_universal_data(self) -> dict:
        try:
            el = self.page.query_selector(
                'script#__UNIVERSAL_DATA_FOR_REHYDRATION__'
            )
            if el:
                return json.loads(el.inner_text())
        except Exception:
            pass
        return {}

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

    def _text(self, selector: str) -> str:
        try:
            el = self.page.query_selector(selector)
            if el:
                return (el.inner_text() or "").strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _as_int(v) -> int | None:
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _collect_video_ids(self, target: int) -> List[str]:
        seen: List[str] = []
        stale = 0
        for _ in range(20):
            hrefs = self.page.eval_on_selector_all(
                'a[href*="/video/"]',
                "els => els.map(e => e.getAttribute('href'))",
            )
            for h in hrefs or []:
                m = re.search(r"/video/(\d+)", h or "")
                if m and m.group(1) not in seen:
                    seen.append(m.group(1))
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

    def _scrape_video(self, username: str, video_id: str) -> Post | None:
        url = f"{self.BASE_URL}/@{username}/video/{video_id}"
        try:
            self.page.goto(url, timeout=45000, wait_until="domcontentloaded")
            self.page.wait_for_timeout(2000)
        except PWTimeout:
            return Post(post_id=video_id, url=url, media_type="video")

        post = Post(post_id=video_id, url=url, media_type="video")

        data = self._read_universal_data()
        item = {}
        if data:
            try:
                item = (
                    data.get("__DEFAULT_SCOPE__", {})
                    .get("webapp.video-detail", {})
                    .get("itemInfo", {})
                    .get("itemStruct", {})
                ) or {}
            except Exception:
                item = {}

        if item:
            post.caption = item.get("desc", "") or ""
            stats = item.get("stats") or item.get("statsV2") or {}
            post.likes = self._as_int(stats.get("diggCount"))
            post.comments = self._as_int(stats.get("commentCount"))
            post.views = self._as_int(stats.get("playCount"))
            ts = item.get("createTime")
            if ts:
                try:
                    from datetime import datetime, timezone
                    post.timestamp = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                except Exception:
                    post.timestamp = str(ts)
            video = item.get("video") or {}
            post.thumbnail_url = video.get("cover") or video.get("dynamicCover") or ""
        else:
            post.caption = self._text('[data-e2e="browse-video-desc"]') or self._meta("og:description")
            post.likes = self.parse_count(self._text('[data-e2e="like-count"]'))
            post.comments = self.parse_count(self._text('[data-e2e="comment-count"]'))
            post.thumbnail_url = self._meta("og:image") or ""

        return post
