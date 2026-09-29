"""Honest-answer bench over HTTP against a running Nekomimi server (Colab ngrok).

Plays each character card through ``/api/nekomimi/start``, ``/answer`` and
``/guess`` until the engine ends the round, then saves the round's
``/api/nekomimi/trace`` and its timing notes. Ported from the Colab bench
driver, with its measurement bugs fixed:

* the loop ends on the engine's stage, not after 25 steps (guesses were
  counted as steps, so every miss was cut off before its third guess);
* timing notes (``eig_calls``, ``eig_qs``, ``hits``, ``dup``) are read from the
  top level of ``payload["timing"]``, where ``timing.Trace.summary`` puts them;
* series options match the card's work exactly (``_same_work``) and guesses
  by ``same_character``, not substrings ("ff" picked any Final Fantasy, and
  a "YoRHa" name counted as 2B);
* questions and guesses are counted separately.

    python scripts/bench_nekomimi_http.py https://<id>.ngrok-free.app --reps 3 --out bench-out

Cards and the yes/no answer rules are the Colab driver's, unchanged, so a
result is comparable with the 67816c0 run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waifu_engine.names import same_character, series_key  # noqa: E402

HEADERS = {
    "ngrok-skip-browser-warning": "1",
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
}
# A round can never need more requests than this: 20 questions plus 2 per
# wrong guess, and 3 guesses. The bound only stops a runaway server.
MAX_STEPS = 60
# Timing notes worth keeping per turn (see ``timing.note`` call sites).
TIMING_KEYS = ("turn", "cands", "eig", "eig_calls", "eig_qs", "recover", "defer",
               "pin_look", "guess", "found", "hits", "dup", "err")

CHARACTERS = {
    "Rem": {
        "series": "Re:Zero",
        "series_aliases": ["re:zero", "re: zero", "re zero", "starting life in another world"],
        "medium": "anime",
        "aliases": ["rem", "re:zero", "レム"],
        "traits": {
            "hair_color": "blue",
            "eye_color": "blue",
            "female": True,
            "male": False,
            "human": False,
            "anime": True,
            "game": False,
            "comic": False,
            "movie": False,
            "tv": False,
            "maid": True,
            "demon": True,
            "horns": True,
            "wings": False,
            "halo": False,
            "magic": True,
            "weapon": True,
            "flail": True,
            "morningstar": True,
            "short_hair": True,
            "long_hair": False,
            "school": False,
            "student": False,
            "glasses": False,
            "eyepatch": False,
            "robot": False,
            "android": False,
            "twin": True,
            "sister": True,
            "blue": True,
            "blonde": False,
            "white": False,
            "red": False,
            "black": False,
            "brown": False,
            "pink": False,
            "protagonist": False,
            "villain": False,
            "playable": False,
        }
    },
    "2B": {
        "series": "NieR:Automata",
        "series_aliases": ["nier", "automata", "nier:automata", "nier automata"],
        "medium": "game",
        "aliases": ["2b", "yorha", "nier", "automata", "yorha no. 2 type b"],
        "traits": {
            "hair_color": "white",
            "eye_color": "blue",
            "female": True,
            "male": False,
            "human": False,
            "anime": False,
            "game": True,
            "comic": False,
            "movie": False,
            "tv": False,
            "android": True,
            "robot": True,
            "sword": True,
            "katana": True,
            "weapon": True,
            "blindfold": True,
            "visor": True,
            "short_hair": True,
            "long_hair": False,
            "school": False,
            "student": False,
            "glasses": False,
            "eyepatch": False,
            "horns": False,
            "wings": False,
            "halo": False,
            "twin": False,
            "white": True,
            "silver": True,
            "black": False,
            "blue": False,
            "blonde": False,
            "brown": False,
            "red": False,
            "protagonist": True,
            "villain": False,
            "playable": True,
            "soldier": True,
            "fighter": True,
            "scifi": True,
            "sci-fi": True,
            "military": True,
            "post-apocalyptic": True,
            "japanese": True,
        }
    },
    "Makima": {
        "series": "Chainsaw Man",
        "series_aliases": ["chainsaw man", "chainsawman", "csm"],
        "medium": "anime",
        "aliases": ["makima", "chainsaw man", "マキマ"],
        "traits": {
            "hair_color": "red",
            "eye_color": "yellow",
            "female": True,
            "male": False,
            "human": False,
            "anime": True,
            "game": False,
            "comic": False,
            "movie": False,
            "tv": False,
            "devil": True,
            "demon": True,
            "suit": True,
            "tie": True,
            "braid": True,
            "long_hair": True,
            "short_hair": False,
            "school": False,
            "student": False,
            "teenager": False,
            "adult": True,
            "glasses": False,
            "eyepatch": False,
            "robot": False,
            "horns": False,
            "wings": False,
            "halo": False,
            "twin": False,
            "red": True,
            "orange": True,
            "black": False,
            "blonde": False,
            "brown": False,
            "white": False,
            "blue": False,
            "protagonist": False,
            "villain": True,
            "playable": False,
            "healer": False,
            "support": False,
            "tall": False,
        }
    },
    "Megumin": {
        "series": "KonoSuba",
        "series_aliases": ["konosuba", "kono subarashii", "god's blessing on this wonderful world"],
        "medium": "anime",
        "aliases": ["megumin", "konosuba", "めぐみん"],
        "traits": {
            "hair_color": "brown",
            "eye_color": "red",
            "female": True,
            "male": False,
            "human": True,
            "anime": True,
            "game": False,
            "comic": False,
            "movie": False,
            "tv": False,
            "magic": True,
            "wizard": True,
            "witch": True,
            "arch wizard": True,
            "explosion": True,
            "eyepatch": True,
            "hat": True,
            "staff": True,
            "short_hair": True,
            "long_hair": False,
            "school": False,
            "student": False,
            "teenager": True,
            "glasses": False,
            "robot": False,
            "horns": False,
            "wings": False,
            "halo": False,
            "twin": False,
            "brown": True,
            "red": False,
            "black": False,
            "blonde": False,
            "white": False,
            "blue": False,
            "red_eyes": True,
            "sword": False,
            "protagonist": False,
            "villain": False,
            "playable": False,
        }
    },
    "Tifa Lockhart": {
        "series": "Final Fantasy VII",
        "series_aliases": ["final fantasy", "final fantasy vii", "ff7", "ffvii", "ff"],
        "medium": "game",
        "aliases": ["tifa", "tifa lockhart", "final fantasy"],
        "traits": {
            "hair_color": "black",
            "eye_color": "red",
            "female": True,
            "male": False,
            "human": True,
            "anime": False,
            "game": True,
            "comic": False,
            "movie": False,
            "tv": False,
            "martial_arts": True,
            "martial artist": True,
            "fighter": True,
            "soldier": False,
            "fist": True,
            "gloves": True,
            "bartender": True,
            "long_hair": True,
            "short_hair": False,
            "school": False,
            "student": False,
            "teenager": False,
            "adult": True,
            "glasses": False,
            "eyepatch": False,
            "robot": False,
            "horns": False,
            "wings": False,
            "halo": False,
            "twin": False,
            "black": True,
            "brown": False,
            "blonde": False,
            "white": False,
            "blue": False,
            "red": False,
            "sword": False,
            "spear": False,
            "lance": False,
            "military": False,
            "japanese": True,
            "fantasy": True,
            "tall": False,
            "protagonist": False,
            "villain": False,
            "playable": True,
        }
    }
}

# Exact names for guess matching and exact series for the series chip.
# The card aliases above include franchise words, so they are not used.
for _name, (_names, _series) in {
    'Rem': (['Rem'], ['Re:Zero', 'Re:ZERO -Starting Life in Another World-']),
    '2B': (['2B', 'YoRHa No. 2 Type B'], ['NieR:Automata', 'Nier Automata']),
    'Makima': (['Makima'], ['Chainsaw Man']),
    'Megumin': (['Megumin'], ['KonoSuba', "KONOSUBA -God's blessing on this wonderful world!"]),
    'Tifa Lockhart': (['Tifa Lockhart'], ['Final Fantasy VII']),
}.items():
    CHARACTERS[_name]["names"] = _names
    CHARACTERS[_name]["series_names"] = _series


def _yesno(info: dict[str, Any], qid: str, qtext: str) -> str:
    """The Colab driver's first-match regex rules for yes/no (unknown -> no)."""
    char_info = info
    checks = [
        (r'\b(female|woman|girl|lady)\b', char_info["traits"]["female"]),
        (r'\bhuman\b', char_info["traits"].get("human", True)),
        (r'\b(male|man|boy)\b', char_info["traits"]["male"]),
        (r'\b(anime|manga|light novel)\b', char_info["traits"]["anime"]),
        (r'\b(video game|game)\b', char_info["traits"]["game"]),
        (r'\b(robot|android|ai|machine|cyborg)\b', char_info["traits"].get("robot", False) or char_info["traits"].get("android", False)),
        (r'\bmaid\b', char_info["traits"].get("maid", False)),
        (r'\b(magic|spell|witch|wizard)\b', char_info["traits"].get("magic", False)),
        (r'\beyepatch\b', char_info["traits"].get("eyepatch", False)),
        (r'\bglasses\b', char_info["traits"].get("glasses", False)),
        (r'\b(school|high school)\b', char_info["traits"].get("school", False)),
        (r'\bstudent\b', char_info["traits"].get("student", False)),
        (r'\b(teen|teenager)\b', char_info["traits"].get("teenager", False)),
        (r'\badult\b', char_info["traits"].get("adult", False)),
        (r'\bhorns?\b', char_info["traits"].get("horns", False)),
        (r'\bwings?\b', char_info["traits"].get("wings", False)),
        (r'\bhalo\b', char_info["traits"].get("halo", False)),
        (r'\b(blindfold|visor|covered eyes)\b', char_info["traits"].get("blindfold", False)),
        (r'\b(demon|devil|oni|monster)\b', char_info["traits"].get("demon", False) or char_info["traits"].get("devil", False)),
        (r'\b(suit|necktie|tie)\b', char_info["traits"].get("suit", False)),
        (r'\bshort[ -]?hair(ed)?\b', char_info["traits"].get("short_hair", False)),
        (r'\blong[ -]?hair(ed)?\b', char_info["traits"].get("long_hair", False)),
        (r'\bblue hair(ed)?\b', char_info["traits"].get("blue", False)),
        (r'\b(white|silver) hair(ed)?\b', char_info["traits"].get("white", False) or char_info["traits"].get("silver", False)),
        (r'\b(red|orange) hair(ed)?\b', char_info["traits"].get("red", False)),
        (r'\b(black|dark) hair(ed)?\b', char_info["traits"].get("black", False)),
        (r'\bblonde hair(ed)?\b', char_info["traits"].get("blonde", False)),
        (r'\bbrown hair(ed)?\b', char_info["traits"].get("brown", False)),
        (r'\bred eyes\b', char_info["traits"].get("red_eyes", False)),
        (r'\b(sword|blade|katana)\b', char_info["traits"].get("sword", False) or char_info["traits"].get("katana", False)),
        (r'\b(spear|lance)\b', char_info["traits"].get("spear", False)),
        (r'\b(staff|wand)\b', char_info["traits"].get("staff", False)),
        (r'\b(weapon|fight|combat)\b', char_info["traits"].get("weapon", True)),
        (r'\bsoldier\b', char_info["traits"].get("soldier", False)),
        (r'\b(martial|fist|fighter)\b', char_info["traits"].get("martial_arts", False) or char_info["traits"].get("fighter", False)),
        (r'\btwin\b', char_info["traits"].get("twin", False)),
        (r'\b(protagonist|lead character|main character)\b', char_info["traits"].get("protagonist", False)),
        (r'\b(villain|antagonist|evil)\b', char_info["traits"].get("villain", False)),
        (r'\bplayable\b', char_info["traits"].get("playable", False)),
        (r'\b(healer|support)\b', char_info["traits"].get("healer", False)),
        (r'\btall\b', char_info["traits"].get("tall", False)),
        (r'\b(japanese|from japan)\b', char_info["traits"].get("japanese", True)),
        (r'\b(post-apocalyptic|at war)\b', char_info["traits"].get("post-apocalyptic", False)),
        (r'\b(sci-fi|science fiction)\b', char_info["traits"].get("scifi", False)),
        (r'\b(military|army)\b', char_info["traits"].get("military", False)),
        (r'\bfantasy\b', char_info["traits"].get("fantasy", False)),
        (r'\bworld-famous\b', False),
        (r'\bschool uniform\b', False),
        (r'\btail\b', False),
        (r'\banimal features\b', False),
        (r'\belf\b', False),
        (r'\bmech\b', False),
        (r'\broyalty\b', False),
        (r'\bsibling\b', False),
        (r'\bsidekick\b', False),
        (r'\brival\b', False),
        (r'\borphan\b', False),
        (r'\bcenturies old\b', False),
        (r'\bstoic\b', False),
        (r'\bcheerful\b', False),
        (r'\bsilent\b', False),
        (r'\bponytail\b', False),
        (r'\btwintails\b', False),
    ]
    
    for pattern, val in checks:
        if re.search(pattern, qtext, re.IGNORECASE) or re.search(pattern, qid, re.IGNORECASE):
            return "yes" if val else "no"
    return "no"


def _same_work(a: str, b: str) -> bool:
    """Whether two series labels name one work, numbered entries kept apart.

    ``names.same_series_key`` is a prefix match, so "Final Fantasy VIII"
    passes for "Final Fantasy VII"; an honest bench must not say so. A
    longer title still matches when the extra part is a subtitle, not an
    entry number ("Final Fantasy VII: Advent Children").
    """
    ka, kb = series_key(a), series_key(b)
    if not ka or not kb:
        return False
    short, long_ = sorted((ka, kb), key=len)
    if not long_.startswith(short):
        return False
    rest = long_[len(short):]
    return not rest or not re.match(r"[ivx0-9]", rest)


def _call(base: str, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """POST ``data`` (or GET when None) and return the JSON reply."""
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(base + path, data=body, headers=HEADERS,
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _options(question: dict[str, Any]) -> dict[str, str]:
    """Option key -> label, from either payload shape."""
    raw = question.get("options") or []
    if isinstance(raw, dict):
        return {k: (v.get("label", k) if isinstance(v, dict) else str(v)) for k, v in raw.items()}
    return {o["key"]: o.get("label", o["key"]) for o in raw if isinstance(o, dict)}


def decide_answer(info: dict[str, Any], question: dict[str, Any]) -> str:
    """The card's honest answer. Series options match the card's work exactly."""
    qid = question.get("qid") or question.get("id") or ""
    qtext = question.get("text") or ""
    options = _options(question)
    if options:
        if qid.startswith("series") or "Which series" in qtext:
            for key, label in options.items():
                if key != "other" and any(_same_work(label, s) for s in info["series_names"]):
                    return key
            return "other" if "other" in options else next(iter(options))
        if qid == "medium":
            med = info["medium"]
            return med if med in options else ("other" if "other" in options else next(iter(options)))
        if "hair" in qid:
            color = info["traits"].get("hair_color", "")
            for key, label in options.items():
                if color and (color in label.lower() or color in key.lower()):
                    return key
            return "other" if "other" in options else next(iter(options))
        for key, label in options.items():
            for trait, value in info["traits"].items():
                if value is True and trait in label.lower():
                    return key
        return "other" if "other" in options else next(iter(options))
    return _yesno(info, qid, qtext)


def _timing_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """This turn's timing notes and the ``pick.eig`` / ``laya.*`` spans."""
    timing = payload.get("timing") or {}
    out = {k: timing[k] for k in TIMING_KEYS if k in timing}
    for name, span in (timing.get("spans") or {}).items():
        if name == "pick.eig" or name.startswith("laya."):
            out[name] = span
    return out


def play(base: str, name: str, info: dict[str, Any]) -> dict[str, Any]:
    """Play one round to the engine's end; return the result and its trace."""
    state = _call(base, "/api/nekomimi/start", {"seed": ""})
    sid = state.get("session_id")
    log: list[dict[str, Any]] = []
    guesses: list[dict[str, Any]] = []
    won = False
    for _step in range(MAX_STEPS):
        stage = state.get("stage")
        timing = _timing_fields(state)
        if stage == "asking":
            question = state.get("question") or {}
            answer = decide_answer(info, question)
            log.append({"qid": question.get("qid") or question.get("id"), "answer": answer,
                        "timing": timing})
            state = _call(base, "/api/nekomimi/answer", {"session_id": sid, "answer": answer})
        elif stage == "guessing":
            guess = state.get("guess") or {}
            correct = any(same_character(guess.get("name") or "", n) for n in info["names"])
            guesses.append({"name": guess.get("name"), "series": guess.get("series"),
                            "correct": correct, "timing": timing})
            print(f"  guess {len(guesses)}: {guess.get('name')} -> {'HIT' if correct else 'miss'}")
            state = _call(base, "/api/nekomimi/guess", {"session_id": sid, "correct": correct})
            if correct:
                won = True
                break
        else:
            break
    trace = _call(base, f"/api/nekomimi/trace/{sid}?target={urllib.parse.quote(name)}")
    eig = [t["timing"]["pick.eig"]["ms"] for t in log if "pick.eig" in t["timing"]]
    return {
        "character": name,
        "won": won,
        "questions": len(log),
        "guesses": guesses,
        "end_stage": state.get("stage"),
        "pick_eig_ms_mean": round(sum(eig) / len(eig), 1) if eig else None,
        "session_id": sid,
        "turns": log,
        "trace": trace,
    }


def main() -> None:
    """Run every card ``--reps`` times and write one JSON per round plus a summary."""
    parser = argparse.ArgumentParser(description="Nekomimi honest-answer HTTP bench")
    parser.add_argument("base_url")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default="bench-out")
    parser.add_argument("--only", default="", help="comma list of card names")
    args = parser.parse_args()
    base = args.base_url.rstrip("/").removesuffix("/nekomimi")
    os.makedirs(args.out, exist_ok=True)
    print("healthz:", _call(base, "/healthz"))
    only = {n.strip().lower() for n in args.only.split(",") if n.strip()}
    summary = []
    for name, info in CHARACTERS.items():
        if only and name.lower() not in only:
            continue
        for rep in range(1, args.reps + 1):
            print(f"{name} #{rep}")
            result = play(base, name, info)
            path = os.path.join(args.out, f"{name.lower().replace(' ', '_')}_{rep}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            summary.append({k: result[k] for k in ("character", "won", "questions",
                                                   "end_stage", "pick_eig_ms_mean")}
                           | {"guesses": [g["name"] for g in result["guesses"]]})
            print(f"  {'WON' if result['won'] else 'lost'} after {result['questions']} "
                  f"questions, {len(result['guesses'])} guesses")
            time.sleep(1)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    wins = sum(r["won"] for r in summary)
    print(f"\n{wins}/{len(summary)} rounds won")


if __name__ == "__main__":
    main()
