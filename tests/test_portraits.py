"""Portrait backfill: AniList art for a gacha name, and no face from the wrong series."""

from waifu_engine.sources import portraits


HOSHINO = {
    "name": "Hoshino Takanashi",
    "series": "Blue Archive The Animation",
    "image_url": "https://s4.anilist.co/file/anilistcdn/character/large/b264242-PISUyKo5o6xr.png",
}
AI = {
    "name": "Ai Hoshino",
    "series": "OSHI NO KO",
    "image_url": "https://s4.anilist.co/file/anilistcdn/character/large/b172759-cccVhJ2fQA92.png",
}


def test_blue_archive_hoshino_gets_the_anilist_portrait(monkeypatch):
    """The guess name is the full name. AniList has the art; Wikipedia does not."""
    monkeypatch.setenv("WAIFU_PORTRAITS", "1")

    def search(query, limit=5):
        assert "Hoshino Takanashi" in query
        return [HOSHINO]

    monkeypatch.setattr("waifu_engine.sources.anilist.search_characters", search)
    monkeypatch.setattr("waifu_engine.sources.wikipedia.search_characters", lambda *a, **k: [])
    row = {"name": "Hoshino Takanashi", "series": "Blue Archive", "image_url": None}
    assert portraits.fill_portraits([row]) == 1
    assert row["image_url"] == HOSHINO["image_url"]


def test_a_shared_token_does_not_steal_another_series_face(monkeypatch):
    """Searching "Hoshino" must not attach Ai Hoshino when the series is Blue Archive."""
    monkeypatch.setenv("WAIFU_PORTRAITS", "1")
    monkeypatch.setattr("waifu_engine.sources.anilist.search_characters", lambda *a, **k: [AI])
    monkeypatch.setattr("waifu_engine.sources.wikipedia.search_characters", lambda *a, **k: [])
    monkeypatch.setattr(portraits, "_from_source_page", lambda *a, **k: None)
    monkeypatch.setattr(portraits, "_from_ddg", lambda *a, **k: None)
    row = {"name": "Hoshino", "series": "Blue Archive", "image_url": None, "source_url": ""}
    assert portraits.fill_portraits([row]) == 0
    assert row["image_url"] is None


def test_wikipedia_thumbnail_fills_when_anilist_has_nobody(monkeypatch):
    """Abby-style pages: no AniList row, but the Wikipedia page image is usable."""
    monkeypatch.setenv("WAIFU_PORTRAITS", "1")
    monkeypatch.setattr("waifu_engine.sources.anilist.search_characters", lambda *a, **k: [])
    monkeypatch.setattr(
        "waifu_engine.sources.wikipedia.search_characters",
        lambda *a, **k: [{
            "name": "Abby",
            "series": "The Last of Us",
            "image_url": "http://upload.wikimedia.org/wikipedia/en/thumb/abby.jpg",
        }],
    )
    row = {"name": "Abby", "series": "The Last of Us", "image_url": None}
    assert portraits.fill_portraits([row]) == 1
    assert row["image_url"].startswith("https://upload.wikimedia.org/")


def test_fair_use_page_uses_the_rest_summary(monkeypatch):
    """pageimages leaves Abby null. The REST summary still has the lead file."""
    monkeypatch.setenv("WAIFU_PORTRAITS", "1")
    monkeypatch.setattr("waifu_engine.sources.anilist.search_characters", lambda *a, **k: [])
    monkeypatch.setattr(
        "waifu_engine.sources.wikipedia.search_characters",
        lambda *a, **k: [{
            "name": "Abby",
            "series": "The Last of Us",
            "image_url": None,
            "source_url": "https://en.wikipedia.org/wiki/Abby_(The_Last_of_Us)",
        }],
    )

    def summary(url, params, accept="application/json"):
        assert "Abby_(The_Last_of_Us)" in url
        return {
            "originalimage": {
                "source": "https://upload.wikimedia.org/wikipedia/en/b/b8/Abby_in_The_Last_of_Us_Part_II.png",
            }
        }

    monkeypatch.setattr("waifu_engine.sources._http.get_json", summary)
    row = {"name": "Abby", "series": "The Last of Us", "image_url": None, "source_url": ""}
    assert portraits.fill_portraits([row]) == 1
    assert row["image_url"].endswith("Abby_in_The_Last_of_Us_Part_II.png")


def test_a_neighbour_page_does_not_donate_its_face():
    """Rossi's splash must not become Rossina's portrait."""
    assert portraits._page_is_this_person(
        "Rossina Wulfperl Luppino", "Arknights: Endfield", "Rossi", ""
    ) is False
    assert portraits._filename_conflicts(
        "Rossina Wulfperl Luppino",
        "https://endfield.wiki.gg/images/thumb/Rossi_Splash_Art.png/1200px-Rossi_Splash_Art.png",
    ) is True
    assert portraits._page_is_this_person(
        "Rossina Wulfperl Luppino", "Arknights: Endfield", "Rossina", ""
    ) is True


def test_duplicate_search_hit_donates_its_image():
    """The first row wins the slot. A later hit must still be able to give it a face."""
    rows = [{"name": "Hoshino Takanashi", "series": "Blue Archive", "image_url": None}]
    assert portraits.donate_image(rows, HOSHINO) is True
    assert rows[0]["image_url"] == HOSHINO["image_url"]
    assert portraits.donate_image(rows, HOSHINO) is False


def test_portraits_stay_off_in_the_offline_suite(monkeypatch):
    """A guess turn must not open a socket just because the image is null."""
    monkeypatch.setenv("WAIFU_PORTRAITS", "0")
    called = []
    monkeypatch.setattr(
        "waifu_engine.sources.anilist.search_characters",
        lambda *a, **k: called.append(1) or [HOSHINO],
    )
    row = {"name": "Hoshino Takanashi", "series": "Blue Archive", "image_url": None}
    assert portraits.fill_portraits([row]) == 0
    assert called == []
    assert row["image_url"] is None
