# HoaxGuard ID

This Python script utilizes the [Playwright](https://playwright.dev/python/) library to perform web scraping and data extraction from public social media profiles on **Instagram**, **TikTok**, and **Twitter/X**. It is particularly designed for obtaining information about public accounts — their username, display name, bio, follower count, post count, and recent posts/reels with captions, likes, comments, timestamps, and profile image URL.

Inspired by the project layout of [`Google-Maps-Scrapper`](https://github.com/zohaibbashir/Google-Maps-Scrapper).

## Key Features

- **Multi-Platform:** Scrape Instagram, TikTok, and Twitter/X profiles from a single CLI.
- **Profile Extraction:** Pulls username, display name, bio, follower / following / post counts, profile image URL, and verified status.
- **Posts & Reels:** Extracts the N most recent posts/reels/videos/tweets per profile with caption, likes, comments, views (where available), timestamp, and thumbnail URL.
- **Count Parsing:** Normalises `1.2M`, `34K`, `1,234` strings into integers.
- **CSV Export:** Flattens one row per post (with profile metadata repeated) and exports to a clean UTF-8 CSV. Empty columns are dropped automatically.
- **Append Mode:** `--append` adds to an existing CSV without clobbering previous runs.

## Installation

```bash
git clone https://github.com/ariqluth/social-media-scrapper.git
cd social-media-scrapper

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m playwright install chromium
```

## Usage

```bash
python main.py -p <platform> -s "<username-or-url>" -t <total> -o <output.csv>
```

### Arguments

- `-p` / `--platform` — one of `instagram`, `tiktok`, `twitter` (alias `x`).
- `-s` / `--search` — username(s) or profile URL(s). Comma-separate multiple targets. `@` prefix is optional.
- `-t` / `--total` — max posts/reels/tweets to scrape per profile (default: `10`).
- `-o` / `--output` — output CSV path (default: `result.csv`).
- `--append` — append to the output file instead of overwriting.
- `--headful` — show the browser window (useful for debugging or solving CAPTCHAs).
- `-v` / `--verbose` — debug logging.

### Examples

Scrape 20 recent Instagram posts from a profile:

```bash
python main.py -p instagram -s "natgeo" -t 20 -o natgeo.csv
```

Scrape two TikTok accounts and append to an existing file:

```bash
python main.py -p tiktok -s "khaby.lame, charlidamelio" -t 15 -o tiktok_creators.csv --append
```

Scrape 50 tweets from a Twitter/X user with the browser visible:

```bash
python main.py -p x -s "naval" -t 50 -o naval.csv --headful
```

## Output Schema

Each row in the CSV is one post with the parent profile's metadata repeated:

| Column | Description |
| --- | --- |
| `platform` | `instagram` / `tiktok` / `twitter` |
| `username` | Handle without `@` |
| `display_name` | Shown name on the profile |
| `bio` | Profile bio / signature |
| `followers`, `following`, `post_count` | Numeric counts |
| `profile_image_url` | Avatar image URL |
| `profile_url` | Canonical profile URL |
| `verified` | `True` / `False` |
| `post_id` | Shortcode / video id / tweet id |
| `post_url` | Canonical post URL |
| `media_type` | `image` / `reel` / `video` / `tweet` |
| `caption` | Post text / description |
| `likes`, `comments`, `views` | Engagement counts (where available) |
| `timestamp` | ISO 8601 when available |
| `thumbnail_url` | Cover image URL |

If a profile has no posts that were scraped, a single row is still emitted with the post fields left blank.

## Project Layout

```
social-media-scrapper/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── scrapers/
    ├── __init__.py
    ├── base.py         # Profile + Post dataclasses, shared helpers
    ├── instagram.py
    ├── tiktok.py
    └── twitter.py
```

## Notes & Caveats

- **This scrapes public, unauthenticated pages only.** No login, no cookies, no private content.
- All three platforms employ anti-scraping measures. Expect rate limiting, intermittent empty results, and occasional CAPTCHAs — run with `--headful` to debug.
- Instagram heavily gates post data behind login; this scraper reads what the public profile HTML and `og:*` meta tags expose, which is enough for a useful row but won't match the Graph API's detail.
- Twitter/X often requires login to view the full timeline. The scraper will still grab profile header info even when the timeline is gated.
- TikTok exposes a large embedded JSON blob (`__UNIVERSAL_DATA_FOR_REHYDRATION__`) which gives the richest output of the three.
- **Respect the ToS of each platform**, and any applicable laws in your jurisdiction. Use responsibly.

## License

[MIT](LICENSE)
