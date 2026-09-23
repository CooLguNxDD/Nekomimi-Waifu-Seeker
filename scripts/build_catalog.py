"""Build ``data/catalog.json`` -- the warm-start pool for the Nekomimi loop.

Anime and manga come from AniList sorted by favourites; games and comics from
Wikipedia category listings, which are the only key-free source that reliably
returns real characters with real prose.

Re-runnable and idempotent. Never called at request time -- the result is data.

    python scripts/build_catalog.py --anilist-pages 6 --limit 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from waifu_engine.sources import anilist, normalize_name, wikipedia  # noqa: E402

# Categories chosen because they list characters, not franchises or episodes.
GAME_CATEGORIES = [
    "Category:Mario (franchise) characters",
    "Category:The Legend of Zelda characters",
    "Category:Sonic the Hedgehog characters",
    "Category:Final Fantasy characters",
    "Category:Street Fighter characters",
    "Category:Tekken characters",
    "Category:Mortal Kombat characters",
    "Category:Resident Evil characters",
    "Category:Metal Gear characters",
    "Category:Overwatch characters",
    "Category:Pokémon characters",
    # These two are the workhorses -- the per-franchise categories are small.
    "Category:Female characters in video games",
    "Category:Male characters in video games",
    "Category:Fighting game characters",
    "Category:Sega characters",
    "Category:Capcom characters",
]

COMIC_CATEGORIES = [
    "Category:Marvel Comics superheroes",
    "Category:Marvel Comics supervillains",
    "Category:DC Comics superheroes",
    "Category:DC Comics supervillains",
    "Category:Marvel Comics characters",
    "Category:DC Comics characters",
    "Category:Image Comics characters",
    "Category:Dark Horse Comics characters",
]


# Category listings are alphabetical slices, so household names drop out purely
# by where they sort. These are fetched by title so they are always present.
SEED_TITLES = [
    # games
    "Mario", "Luigi", "Princess Peach", "Bowser", "Yoshi", "Donkey Kong",
    "Link (The Legend of Zelda)", "Princess Zelda", "Ganon", "Samus Aran",
    "Kirby (character)", "Pikachu", "Sonic the Hedgehog", "Cloud Strife",
    "Sephiroth (Final Fantasy)", "Tifa Lockhart", "Lara Croft", "Master Chief",
    "Kratos (God of War)", "Solid Snake", "Ryu (Street Fighter)", "Chun-Li",
    "Pac-Man", "Steve (Minecraft)", "Geralt of Rivia", "Aloy (Horizon)",
    "Ellie (The Last of Us)", "2B (Nier: Automata)", "Hatsune Miku",
    "Arthur Morgan", "Sackboy", "Crash Bandicoot", "Mega Man",
    # comics
    "Spider-Man", "Batman", "Superman", "Wonder Woman", "Iron Man",
    "Captain America", "Thor (Marvel Comics)", "Hulk", "Black Widow (Natasha Romanova)",
    "Deadpool", "Wolverine (character)", "Joker (character)", "Harley Quinn",
    "Catwoman", "The Flash", "Aquaman", "Green Lantern", "Doctor Strange",
    "Black Panther (character)", "Magneto (Marvel Comics)", "Venom (Marvel Comics)",
    "Thanos", "Storm (Marvel Comics)", "Jean Grey", "Robin (character)",
    # anime / manga with strong Wikipedia articles
    "Goku", "Naruto Uzumaki", "Monkey D. Luffy", "Sailor Moon (character)",
    "Astro Boy", "Doraemon", "Ash Ketchum", "Light Yagami", "Edward Elric",
]


def _catalog_entry(cand: dict) -> dict:
    """Trim a source candidate down to the catalog contract."""
    return {
        "id": cand["id"],
        "name": cand["name"],
        "series": cand.get("series") or "Unknown",
        "medium": cand.get("medium") or "unknown",
        "tags": cand.get("tags") or [],
        "blurb": (cand.get("blurb") or "")[:700],
        "image_url": cand.get("image_url"),
        "source_url": cand.get("source_url"),
        "popularity": int(cand.get("popularity") or 0),
        "source": cand.get("source", ""),
    }


def collect_anilist(pages: int, per_page: int, log) -> list[dict]:
    out = []
    for page in range(1, pages + 1):
        chars = anilist.top_characters(page=page, per_page=per_page)
        log(f"  anilist page {page}: {len(chars)}")
        if not chars:
            break
        out.extend(chars)
    return out


def collect_wikipedia(categories: list[str], per_category: int, log) -> list[dict]:
    out = []
    for cat in categories:
        titles = wikipedia.category_members(cat, limit=per_category)
        if not titles:
            log(f"  {cat}: no members")
            continue
        pages = wikipedia.pages_by_title(titles[:per_category])
        log(f"  {cat}: {len(titles)} titles -> {len(pages)} characters")
        out.extend(pages)
    return out


def build(args, log=print) -> list[dict]:
    merged: dict[str, dict] = {}

    def take(cands: list[dict]) -> None:
        for cand in cands:
            key = normalize_name(cand.get("name", ""))
            if not key:
                continue
            entry = _catalog_entry(cand)
            prev = merged.get(key)
            # Keep whichever record is better described; popularity is not
            # comparable across sources (AniList favourites vs page views).
            if prev is None or len(entry["blurb"]) > len(prev["blurb"]):
                merged[key] = entry

    log("Wikipedia (must-have icons):")
    seeded = wikipedia.pages_by_title(SEED_TITLES)
    log(f"  {len(SEED_TITLES)} titles -> {len(seeded)} characters")
    take(seeded)
    seed_keys = {normalize_name(c["name"]) for c in seeded}
    log("AniList (anime/manga):")
    take(collect_anilist(args.anilist_pages, args.anilist_per_page, log))
    log("Wikipedia (games):")
    take(collect_wikipedia(GAME_CATEGORIES, args.per_category, log))
    log("Wikipedia (comics):")
    take(collect_wikipedia(COMIC_CATEGORIES, args.per_category, log))

    # The icons are the point of the seed list, so they are never the ones the
    # limit cuts.
    ranked = sorted(merged.items(), key=lambda kv: kv[1]["popularity"], reverse=True)
    keep = [e for k, e in ranked if k in seed_keys]
    keep += [e for k, e in ranked if k not in seed_keys][: max(0, args.limit - len(keep))]
    return sorted(keep, key=lambda c: c["popularity"], reverse=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "catalog.json"))
    ap.add_argument("--anilist-pages", type=int, default=6)
    ap.add_argument("--anilist-per-page", type=int, default=50)
    ap.add_argument("--per-category", type=int, default=40)
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args(argv)

    entries = build(args)
    if not entries:
        print("no entries collected; catalog left untouched", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")

    by_medium: dict[str, int] = {}
    for e in entries:
        by_medium[e["medium"]] = by_medium.get(e["medium"], 0) + 1
    print(f"\nwrote {len(entries)} entries to {out}")
    print("by medium:", dict(sorted(by_medium.items(), key=lambda kv: -kv[1])))
    print("no blurb:", sum(1 for e in entries if not e["blurb"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
