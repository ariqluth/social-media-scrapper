import argparse
import logging
import os
import sys
from typing import List

import pandas as pd
from playwright.sync_api import sync_playwright

from scrapers import SCRAPERS, Profile
meta_data = {
    "name": "dicoding:email",
    "content": "ttzluthfi@gmail.com"
}

print(meta_data["content"]) 

def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_targets(search: str) -> List[str]:
    """Accept 'user1,user2' or whitespace-separated usernames/URLs."""
    if not search:
        return []
    raw = [p.strip() for p in search.replace("\n", ",").split(",")]
    return [r for r in raw if r]


CSV_COLUMNS = [
    "platform", "username", "display_name", "bio",
    "followers", "following", "post_count",
    "profile_image_url", "profile_url", "verified",
    "post_id", "post_url", "media_type",
    "caption", "likes", "comments", "views",
    "timestamp", "thumbnail_url",
]


def save_profiles_to_csv(profiles: List[Profile], output_path: str, append: bool = False) -> None:
    rows = []
    for p in profiles:
        rows.extend(p.flatten())
    if not rows:
        logging.warning("No rows to write.")
        return

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    write_header = True
    mode = "w"
    if append and os.path.exists(output_path):
        mode = "a"
        write_header = False

    df.to_csv(output_path, index=False, mode=mode, header=write_header, encoding="utf-8-sig")
    logging.info("Saved %d rows to %s", len(df), output_path)


def run(platform: str, targets: List[str], total: int, output: str, append: bool, headless: bool) -> None:
    scraper_cls = SCRAPERS.get(platform.lower())
    if not scraper_cls:
        logging.error("Unknown platform '%s'. Choose from: %s", platform, ", ".join(SCRAPERS.keys()))
        sys.exit(2)

    profiles: List[Profile] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 900},
            locale="en-US",
        )
        page = context.new_page()
        scraper = scraper_cls(page)

        for t in targets:
            logging.info("=== Scraping %s: %s ===", platform, t)
            try:
                profile = scraper.scrape_profile(t, max_posts=total)
                profiles.append(profile)
                logging.info(
                    "Done %s (followers=%s, posts scraped=%d)",
                    t, profile.followers, len(profile.posts),
                )
            except Exception as exc:
                logging.exception("Failed to scrape %s: %s", t, exc)

        context.close()
        browser.close()

    save_profiles_to_csv(profiles, output, append=append)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape public profiles + posts from Instagram, TikTok, or Twitter/X."
    )
    parser.add_argument("-p", "--platform", type=str, required=True,
                        choices=list(SCRAPERS.keys()),
                        help="Platform to scrape: instagram | tiktok | twitter | x")
    parser.add_argument("-s", "--search", type=str, required=True,
                        help="Username(s) or profile URL(s). Comma-separate multiple targets.")
    parser.add_argument("-t", "--total", type=int, default=10,
                        help="Max number of posts/reels to scrape per profile (default: 10)")
    parser.add_argument("-o", "--output", type=str, default="result.csv",
                        help="Output CSV file path (default: result.csv)")
    parser.add_argument("--append", action="store_true",
                        help="Append results to the output file instead of overwriting")
    parser.add_argument("--headful", action="store_true",
                        help="Show the browser window (useful for debugging / CAPTCHAs)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    targets = parse_targets(args.search)
    if not targets:
        logging.error("No targets provided in --search.")
        sys.exit(2)

    run(
        platform=args.platform,
        targets=targets,
        total=args.total,
        output=args.output,
        append=args.append,
        headless=not args.headful,
    )


if __name__ == "__main__":
    main()
