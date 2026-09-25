"""Portrait backfill searches by name and series. It does not scrape a wiki page."""

from waifu_engine.sources import portraits


HOSHINO_ART = "https://cdn.example/hoshino-blue-archive.png"
ROSSINA_ART = "https://cdn.example/rossina-endfield.png"


def test_portrait_query_names_the_series():
    """The series is what makes the image search this Hoshino, not another."""
    assert portraits.portrait_query("Hoshino Takanashi", "Blue Archive") == (
        "Hoshino Takanashi Blue Archive character"
    )
    assert portraits.portrait_query(
        "Rossina Wulfperl Luppino", "Arknights: Endfield", "game"
    ) == "Rossina Wulfperl Luppino Arknights: Endfield character"
    assert portraits.portrait_query("Abby", "", "game") == "Abby game character"


def test_missing_image_is_filled_from_the_image_search(monkeypatch):
    """Guess/win rows with a null URL take the first HTTPS hit for that query."""
    monkeypatch.setenv("WAIFU_PORTRAITS", "1")
    seen = []

    def fetch(query, extra=" anime character"):
        seen.append((query, extra))
        if "Rossina" in query:
            return ROSSINA_ART
        return HOSHINO_ART

    monkeypatch.setattr("waifu_engine.web_search.fetch_image_url", fetch)
    rows = [
        {"name": "Hoshino Takanashi", "series": "Blue Archive", "image_url": None},
        {"name": "Rossina Wulfperl Luppino", "series": "Arknights: Endfield", "image_url": ""},
    ]
    assert portraits.fill_portraits(rows) == 2
    assert rows[0]["image_url"] == HOSHINO_ART
    assert rows[1]["image_url"] == ROSSINA_ART
    assert seen[0] == ("Hoshino Takanashi Blue Archive character", "")
    assert "anime character" not in seen[0][0]


def test_a_bare_given_name_is_not_searched(monkeypatch):
    """'Hoshino' alone is several people. No series means no image search."""
    monkeypatch.setenv("WAIFU_PORTRAITS", "1")
    called = []
    monkeypatch.setattr(
        "waifu_engine.web_search.fetch_image_url",
        lambda *a, **k: called.append(1) or HOSHINO_ART,
    )
    row = {"name": "Hoshino", "series": "", "image_url": None}
    assert portraits.fill_portraits([row]) == 0
    assert called == []
    assert row["image_url"] is None


def test_an_existing_https_portrait_is_not_replaced(monkeypatch):
    """A URL already on the row is the portrait. Search is for the gaps."""
    monkeypatch.setenv("WAIFU_PORTRAITS", "1")
    called = []
    monkeypatch.setattr(
        "waifu_engine.web_search.fetch_image_url",
        lambda *a, **k: called.append(1) or ROSSINA_ART,
    )
    row = {"name": "Hoshino Takanashi", "series": "Blue Archive", "image_url": HOSHINO_ART}
    assert portraits.fill_portraits([row]) == 0
    assert called == []
    assert row["image_url"] == HOSHINO_ART


def test_svg_and_http_are_not_kept():
    """Icons and plain HTTP are not a portrait. HTTP is upgraded when it is an image."""
    assert portraits.usable_image("https://cdn.example/icon.svg") is None
    assert portraits.usable_image("http://cdn.example/face.png") == "https://cdn.example/face.png"
    assert portraits.usable_image(None) is None


def test_search_silence_leaves_the_cat(monkeypatch):
    """When the image search returns nothing, the URL stays empty."""
    monkeypatch.setenv("WAIFU_PORTRAITS", "1")
    monkeypatch.setattr("waifu_engine.web_search.fetch_image_url", lambda *a, **k: None)
    row = {
        "name": "Rossina Wulfperl Luppino",
        "series": "Arknights: Endfield",
        "image_url": None,
    }
    assert portraits.fill_portraits([row]) == 0
    assert row["image_url"] is None


def test_duplicate_search_hit_donates_its_image():
    """The first row wins the slot. A later hit must still be able to give it a face."""
    rows = [{"name": "Hoshino Takanashi", "series": "Blue Archive", "image_url": None}]
    incoming = {"name": "Takanashi Hoshino", "image_url": HOSHINO_ART}
    assert portraits.donate_image(rows, incoming) is True
    assert rows[0]["image_url"] == HOSHINO_ART
    assert portraits.donate_image(rows, incoming) is False


def test_portraits_stay_off_in_the_offline_suite(monkeypatch):
    """A guess turn must not open a socket just because the image is null."""
    monkeypatch.setenv("WAIFU_PORTRAITS", "0")
    called = []
    monkeypatch.setattr(
        "waifu_engine.web_search.fetch_image_url",
        lambda *a, **k: called.append(1) or HOSHINO_ART,
    )
    row = {"name": "Hoshino Takanashi", "series": "Blue Archive", "image_url": None}
    assert portraits.fill_portraits([row]) == 0
    assert called == []
    assert row["image_url"] is None
