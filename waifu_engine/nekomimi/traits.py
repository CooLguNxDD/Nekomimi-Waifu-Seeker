"""ACG (Anime / Comic / Game) trait question bank.

Laya cannot write questions -- it only picks among typed options. So every
question the engine can ask lives here, and Laya's job is to choose which one
is worth asking next. Dynamic questions mined from search snippets are built
with the same shape (see ``make_dynamic``).

Yes/no question shape (``_q``)::

    {
      "id": "medium_game",
      "text": "Is your character from a video game?",     # shown to the user
      "instructions": "...",                              # given to Laya per candidate
      "category": "medium",
      "tags_true": [...],   # candidate tags that imply the answer is yes
      "tags_false": [...],  # candidate tags that imply the answer is no
      "prior": 0.35,        # rough P(yes) across all ACG characters
      "kind": "yesno",
    }

Multiple-choice questions (``_choice``) carry ``"kind": "choice"`` and an
``options`` map; the player answers with an option key.
"""

from __future__ import annotations

import re
from typing import Any

# A "detail" answer adds free text to the session constraints instead of
# scoring the current question, so it carries no evidence weight of its own.
ANSWER_WEIGHT: dict[str, float] = {"yes": 1.0, "no": -1.0, "detail": 0.0}
ANSWERS = tuple(ANSWER_WEIGHT)


_LEAD = re.compile(
    r"^(is|does|did|has|are|do)\s+(the\s+)?(character\s+)?"
    r"(described\s+)?(in\s+`candidate`|`candidate`)\s*",
    re.I,
)


def noul_criteria(instructions: str, detail: str | None = None) -> dict[str, str]:
    """Explicit true/false option text for a ``noul`` question.

    Without this Laya scores against its generic "yes, the statement holds",
    which is a much weaker target than a restatement of the actual claim.
    """
    clause = (detail or "").strip() or _LEAD.sub("", instructions.strip()).rstrip("?").strip()
    clause = clause[0].lower() + clause[1:] if clause else "the statement holds"
    # Framed rather than inflected: the clause can be adjectival ("female") or
    # verbal ("die at some point"), and "the character is die" reads as noise.
    return {
        "true": f"yes, this is true of the character: {clause}",
        "false": f"no, this is not true of the character: {clause}",
    }


def _q(
    qid: str,
    category: str,
    text: str,
    instructions: str,
    criteria_detail: str | None = None,
    tags_true: list[str] | None = None,
    tags_false: list[str] | None = None,
    prior: float = 0.5,
) -> dict[str, Any]:
    """Keep ``instructions`` short.

    Measured on the typed-decisions checkpoint, the same question asked with a
    140-character instruction scored Mario 0.38 for "is this a video game
    character"; asked in 50 characters it scored 0.71. Elaboration belongs in
    ``criteria_detail``, which only shapes the true/false option text.
    """
    return {
        "criteria": noul_criteria(instructions, criteria_detail),
        "id": qid,
        "category": category,
        "text": text,
        "instructions": instructions,
        "tags_true": tags_true or [],
        "tags_false": tags_false or [],
        "prior": prior,
        "dynamic": False,
        "kind": "yesno",
    }


def _choice(
    qid: str,
    category: str,
    text: str,
    instructions: str,
    options: list[tuple[str, str, str, list[str], str, float]],
) -> dict[str, Any]:
    """A multiple-choice question, scored with one Laya ``choice`` per candidate.

    options: (key, user-facing label, Laya option text, candidate tags that
    imply this option, search keyword fact or "" for none, prior). Keep it to
    eight options at most: Laya is weak on wide choice sets and every option
    string counts against ``head_max_len``.
    """
    assert 2 <= len(options) <= 8, qid
    return {
        "kind": "choice",
        "id": qid,
        "category": category,
        "text": text,
        "instructions": instructions,
        "criteria": {key: f"the character has {crit}" for key, _, crit, _, _, _ in options},
        "options": {
            key: {"label": label, "tags": list(tags), "fact": fact, "prior": prior}
            for key, label, _, tags, fact, prior in options
        },
        # Kept for code that treats every question alike; choice evidence
        # never reads them.
        "tags_true": [],
        "tags_false": [],
        "prior": 0.5,
        "dynamic": False,
    }


def is_choice(question: dict[str, Any]) -> bool:
    return question.get("kind") == "choice"


def valid_answers(question: dict[str, Any]) -> tuple[str, ...]:
    """Answers accepted for ``question``: option keys for choice, else yes/no."""
    if is_choice(question):
        return (*question["options"], "detail")
    return ANSWERS


def _trait_block(
    category: str,
    prefix: str,
    items: list[tuple[str, str, str, float]],
    tag_prefix: str = "",
) -> list[dict[str, Any]]:
    """items: (slug, user-facing phrase, Laya instruction clause, prior).

    ``tag_prefix`` namespaces the tag so two blocks cannot share a slug -- a
    bare "blue" would otherwise satisfy both the hair and the eye question.
    """
    out = []
    for slug, label, clause, prior in items:
        out.append(
            _q(
                f"{prefix}_{slug}",
                category,
                f"Is your character {label}?",
                f"Is the character in `candidate` {clause}?",
                tags_true=[f"{tag_prefix}{slug}"],
                prior=prior,
            )
        )
    return out


MEDIUM_VALUES = ("anime", "manga", "comic", "game")


def clue_question(qid: str, clues: str) -> dict[str, Any]:
    """Evaluate a user's free-text search clue against one retrieved identity."""
    question = _q(qid, "clue", "Does your character match these clues?",
                  "Does `candidate` match the supplied `clues`?")
    question["clues"] = clues
    return question

QUESTION_BANK: list[dict[str, Any]] = [
    # --- medium: highest information gain, asked first ---
    _q("medium_game", "medium", "Is your character from a video game?",
       "Is the character in `candidate` from a video game?", criteria_detail="from a video game, visual novel or gacha game -- not anime, manga or comics",
       tags_true=["game", "vn", "rpg", "gacha", "fighting-game"],
       tags_false=["anime", "manga", "comic"], prior=0.35),
    _q("medium_anime", "medium", "Is your character from an anime or manga?",
       "Is the character in `candidate` from anime or manga?", criteria_detail="from Japanese anime, manga or a light novel",
       tags_true=["anime", "manga", "light-novel"],
       tags_false=["comic", "game"], prior=0.45),
    _q("medium_comic", "medium", "Is your character from a Western comic?",
       "Is the character in `candidate` from a Western comic?", criteria_detail="from a Western comic book, graphic novel or webtoon such as Marvel or DC",
       tags_true=["comic", "marvel", "dc", "webtoon"],
       tags_false=["anime", "manga", "game"], prior=0.2),
    _q("medium_vn", "medium", "Is your character from a visual novel or dating sim?",
       "Is the character in `candidate` from a visual novel?", criteria_detail="from a visual novel, dating sim or otome game",
       tags_true=["vn", "otome", "dating-sim"], prior=0.1),
    _q("medium_gacha", "medium", "Is your character from a gacha or live-service mobile game?",
       "Is the character in `candidate` from a gacha mobile game?", criteria_detail="from a gacha or live-service mobile game such as Genshin Impact, Fate/Grand Order or Blue Archive",
       tags_true=["gacha", "mobile"], prior=0.12),
    _q("medium_fighting", "medium", "Is your character on a fighting game roster?",
       "Is the character in `candidate` a fighting game fighter?", criteria_detail="a playable fighter in a series such as Street Fighter, Tekken, Guilty Gear or Smash Bros",
       tags_true=["fighting-game"], prior=0.08),
    _q("medium_playable", "medium", "Is your character playable by the user?",
       "Is the character in `candidate` playable?", criteria_detail="a player-controlled or playable character in their source work",
       tags_true=["playable"], prior=0.3),

    # --- identity ---
    _q("gender_female", "gender", "Is your character female?",
       "Is the character in `candidate` female?", tags_true=["female"], tags_false=["male"], prior=0.5),
    _q("gender_male", "gender", "Is your character male?",
       "Is the character in `candidate` male?", tags_true=["male"], tags_false=["female"], prior=0.5),
    _q("gender_ambiguous", "gender", "Is your character's gender ambiguous or non-binary?",
       "Is the character in `candidate` androgynous?", criteria_detail="gender-ambiguous, androgynous, non-binary, or known for crossdressing",
       tags_true=["androgynous"], prior=0.06),

    _q("species_human", "species", "Is your character human?",
       "Is the character in `candidate` human?", criteria_detail="a normal human being, not a demon, god, robot, alien, beast or spirit",
       tags_true=["human"], tags_false=["demon", "robot", "alien", "beast", "god"], prior=0.55),
    _q("species_robot", "species", "Is your character a robot, android or AI?",
       "Is the character in `candidate` a robot, android, cyborg or artificial intelligence?",
       tags_true=["robot", "android", "ai"], prior=0.1),
    _q("species_demon", "species", "Is your character a demon, devil or monster?",
       "Is the character in `candidate` a demon, devil, oni, vampire or monster?",
       tags_true=["demon", "vampire", "monster"], prior=0.12),
    _q("species_god", "species", "Is your character a god or deity?",
       "Is the character in `candidate` a god, goddess, deity or divine being?",
       tags_true=["god", "deity"], prior=0.07),
    _q("species_beast", "species", "Does your character have animal features?",
       "Does the character in `candidate` have animal features?", criteria_detail="has animal ears, a tail, or is an anthropomorphic beastkin",
       tags_true=["beast", "kemonomimi", "furry"], prior=0.12),
    _q("species_alien", "species", "Is your character an alien or from another world?",
       "Is the character in `candidate` an alien, extraterrestrial, or a native of another world or dimension?",
       tags_true=["alien", "isekai"], prior=0.12),
    _q("species_undead", "species", "Is your character undead or a ghost?",
       "Is the character in `candidate` undead, a ghost, a spirit or a revenant?",
       tags_true=["undead", "ghost"], prior=0.06),

    # --- narrative role ---
    _q("role_protagonist", "role", "Is your character the main protagonist?",
       "Is the character in `candidate` the main protagonist or lead of their work?",
       tags_true=["protagonist", "main"], tags_false=["antagonist", "support"], prior=0.3),
    _q("role_antagonist", "role", "Is your character a villain or antagonist?",
       "Is the character in `candidate` a villain, antagonist or main enemy?",
       tags_true=["antagonist", "villain"], tags_false=["protagonist"], prior=0.22),
    _q("role_antihero", "role", "Is your character an anti-hero?",
       "Is the character in `candidate` an anti-hero?", criteria_detail="morally grey, fighting for good by questionable means",
       tags_true=["antihero"], prior=0.14),
    _q("role_love_interest", "role", "Is your character a canonical love interest?",
       "Is the character in `candidate` a canonical love interest?", criteria_detail="a canonical romantic partner or love interest of another major character",
       tags_true=["love-interest", "romance"], prior=0.22),
    _q("role_rival", "role", "Is your character the protagonist's rival?",
       "Is the character in `candidate` a rival or foil to the protagonist?",
       tags_true=["rival"], prior=0.12),
    _q("role_mentor", "role", "Is your character a mentor or teacher figure?",
       "Is the character in `candidate` a mentor, master or teacher to the protagonist?",
       tags_true=["mentor", "teacher"], prior=0.1),
    _q("role_sidekick", "role", "Is your character a sidekick or supporting cast?",
       "Is the character in `candidate` a sidekick or supporting-cast member rather than a lead?",
       tags_true=["support", "sidekick"], prior=0.35),
    _q("role_dies", "role", "Does your character die in their story?",
       "Does the character in `candidate` die at some point in their source work?",
       tags_true=["dies"], prior=0.2),
    _q("role_leader", "role", "Does your character lead a group or organization?",
       "Does the character in `candidate` lead a team, guild, crew, squad or organization?",
       tags_true=["leader", "captain"], prior=0.18),

    # --- appearance: hair ---
    # Hair and eye colour are mutually exclusive, so they are single
    # multiple-choice questions scored with a Laya ``choice`` per candidate
    # rather than a run of yes/no questions. Option tags keep the bare hair
    # slugs so web_search.mine_trait_slugs colours line up.
    _choice("hair_color", "hair_color", "What colour is your character's hair?",
            "What hair colour does the character in `candidate` have?", [
        ("blonde", "Blonde", "blonde, golden or yellow hair", ["blonde"], "blonde hair", 0.18),
        ("black", "Black", "black hair", ["black"], "black hair", 0.2),
        ("brown", "Brown", "brown hair", ["brown"], "brown hair", 0.16),
        ("white", "White / silver", "white, silver or grey hair", ["white"], "silver hair", 0.12),
        ("red", "Red / orange", "red, crimson or orange hair", ["red"], "red hair", 0.1),
        ("blue", "Blue", "blue hair", ["blue"], "blue hair", 0.08),
        ("pink", "Pink", "pink hair", ["pink"], "pink hair", 0.06),
        ("other", "Other (green, purple...)", "green, purple or another hair colour",
         ["green", "purple"], "", 0.1),
    ]),
    *_trait_block("hair", "hair", [
        ("long", "long-haired", "described as having long hair", 0.4),
        ("short", "short-haired", "described as having short hair", 0.35),
        ("twintails", "wearing twintails", "known for wearing twintails or pigtails", 0.08),
        ("ponytail", "wearing a ponytail", "known for wearing a ponytail", 0.1),
    ]),

    # --- appearance: eyes ---
    # Slugs are prefixed: a bare "blue" would collide with the hair colour and
    # make one tag satisfy both a hair and an eye question.
    _choice("eye_color", "eye_color", "What colour are your character's eyes?",
            "What eye colour does the character in `candidate` have?", [
        ("blue", "Blue", "blue eyes", ["eyes-blue"], "blue eyes", 0.2),
        ("red", "Red", "red or crimson eyes", ["eyes-red"], "red eyes", 0.1),
        ("green", "Green", "green eyes", ["eyes-green"], "green eyes", 0.1),
        ("gold", "Gold / amber", "golden, yellow or amber eyes", ["eyes-gold"], "golden eyes", 0.1),
        ("brown", "Brown / dark", "brown or dark eyes", ["eyes-brown"], "", 0.3),
        ("other", "Other", "another eye colour, such as purple or pink", [], "", 0.2),
    ]),
    *_trait_block("eyes", "eyes", [
        ("heterochromia", "heterochromatic (two eye colours)", "having two different eye colours", 0.03),
    ], tag_prefix="eyes-"),

    # --- appearance: other ---
    _q("look_glasses", "look", "Does your character wear glasses?",
       "Is the character in `candidate` regularly shown wearing glasses?", tags_true=["glasses"], prior=0.12),
    _q("look_mask", "look", "Does your character wear a mask or helmet?",
       "Does the character in `candidate` regularly wear a mask, helmet or face covering?",
       tags_true=["mask", "helmet"], prior=0.08),
    _q("look_armor", "look", "Does your character wear armor?",
       "Does the character in `candidate` wear armour or a combat suit as their usual outfit?",
       tags_true=["armor", "knight"], prior=0.12),
    _q("look_uniform", "look", "Does your character wear a school uniform?",
       "Does the character in `candidate` usually wear a school uniform?", tags_true=["uniform", "student"], prior=0.16),
    _q("look_tall", "look", "Is your character notably tall?",
       "Is the character in `candidate` described as notably tall?", tags_true=["tall"], prior=0.18),
    _q("look_short", "look", "Is your character notably short or small?",
       "Is the character in `candidate` described as notably short, small or child-sized?",
       tags_true=["small", "chibi"], prior=0.16),
    _q("look_scar", "look", "Does your character have a visible scar or tattoo?",
       "Does the character in `candidate` have a signature visible scar, tattoo or marking?",
       tags_true=["scar", "tattoo"], prior=0.12),
    _q("look_horns", "look", "Does your character have horns or wings?",
       "Does the character in `candidate` have horns, wings or a halo?", tags_true=["horns", "wings"], prior=0.08),

    # --- age ---
    _q("age_child", "age", "Is your character a child?",
       "Is the character in `candidate` a child, under roughly 13 years old?", tags_true=["child"], prior=0.1),
    _q("age_teen", "age", "Is your character a teenager?",
       "Is the character in `candidate` a teenager, roughly 13-19 years old?", tags_true=["teen", "student"], prior=0.35),
    _q("age_adult", "age", "Is your character an adult?",
       "Is the character in `candidate` an adult, 20 years or older?", tags_true=["adult"], prior=0.45),
    _q("age_ancient", "age", "Is your character centuries old?",
       "Is the character in `candidate` centuries old?", criteria_detail="centuries old, immortal, or ageless despite a young appearance",
       tags_true=["immortal", "ancient"], prior=0.1),

    # --- personality archetypes ---
    *_trait_block("pers", "pers", [
        ("tsundere", "a tsundere", "a tsundere", 0.1),
        ("kuudere", "cold and emotionless", "cold, blunt and emotionless", 0.08),
        ("yandere", "obsessive and dangerous in love", "obsessively and dangerously devoted to someone", 0.05),
        ("dandere", "shy and quiet", "shy, timid or very quiet around others", 0.1),
        ("genki", "energetic and cheerful", "energetic, loud and relentlessly cheerful", 0.16),
        ("stoic", "stoic and serious", "stoic, serious and composed", 0.2),
        ("arrogant", "arrogant or prideful", "arrogant, prideful or boastful", 0.14),
        ("kind", "kind and gentle", "kind, gentle and caring towards others", 0.3),
        ("cunning", "cunning or manipulative", "cunning, scheming or manipulative", 0.15),
        ("comic", "comic relief", "used as comic relief in their work", 0.12),
        ("mysterious", "mysterious", "mysterious, with a hidden past or unclear motives", 0.18),
        ("lazy", "lazy or unmotivated", "lazy, apathetic or unmotivated", 0.08),
        ("hotblooded", "hot-blooded", "hot-blooded, impulsive and quick to fight", 0.16),
    ]),

    # --- occupation / class ---
    *_trait_block("job", "job", [
        ("student", "a student", "a student at a school, academy or university", 0.28),
        ("soldier", "a soldier or fighter", "a soldier, knight, mercenary or professional fighter", 0.25),
        ("mage", "a magic user", "a mage, wizard, witch or spellcaster", 0.18),
        ("ninja", "a ninja or assassin", "a ninja, shinobi or assassin", 0.08),
        ("pirate", "a pirate", "a pirate", 0.04),
        ("detective", "a detective or investigator", "a detective, investigator or police officer", 0.06),
        ("doctor", "a doctor or scientist", "a doctor, medic, scientist or researcher", 0.08),
        ("idol", "an idol, singer or musician", "an idol, singer, musician or performer", 0.07),
        ("royalty", "royalty or nobility", "royalty, nobility or an aristocrat", 0.1),
        ("maid", "a maid or butler", "a maid, butler or household servant", 0.05),
        ("priest", "a priest, nun or shrine maiden", "a priest, nun, monk or shrine maiden", 0.06),
        ("chef", "a chef or cook", "a chef, cook or restaurateur", 0.03),
        ("pilot", "a pilot or mech user", "a pilot of a mech, plane or spacecraft", 0.06),
        ("hacker", "a hacker or engineer", "a hacker, programmer or engineer", 0.05),
        ("athlete", "an athlete", "an athlete or professional sports player", 0.06),
    ]),

    # --- powers / combat ---
    *_trait_block("power", "power", [
        ("magic", "able to use magic", "able to use magic or supernatural spells", 0.3),
        ("superhuman", "superhumanly strong", "superhumanly strong or durable", 0.3),
        ("sword", "a sword user", "a character whose signature weapon is a sword or blade", 0.25),
        ("gun", "a gun user", "a character whose signature weapon is a gun or firearm", 0.18),
        ("bow", "an archer", "an archer or bow user", 0.05),
        ("fists", "an unarmed fighter", "a hand-to-hand or martial arts fighter", 0.2),
        ("elemental", "able to control an element", "able to control an element", 0.22),
        ("psychic", "psychic or telepathic", "psychic, telepathic or telekinetic", 0.08),
        ("healer", "a healer or support fighter", "a healer or support-type fighter", 0.1),
        ("transform", "able to transform", "able to transform into a stronger or alternate form", 0.22),
        ("flight", "able to fly", "able to fly under their own power", 0.15),
        ("noncombat", "not a fighter at all", "a non-combatant who does not fight", 0.15),
    ]),

    # --- franchise / meta ---
    _q("fame_famous", "fame", "Is your character world-famous outside their fandom?",
       "Is the character in `candidate` famous to the general public?", criteria_detail="recognisable even to people who do not follow the source work",
       tags_true=["famous", "iconic"], prior=0.2),
    _q("fame_mascot", "fame", "Is your character the mascot or face of their franchise?",
       "Is the character in `candidate` the mascot, cover character or face of their franchise?",
       tags_true=["mascot"], prior=0.12),
    _q("era_old", "era", "Did your character first appear before the year 2000?",
       "Did the character in `candidate` first appear before the year 2000?", tags_true=["retro"], prior=0.25),
    _q("era_recent", "era", "Did your character first appear in the last ten years?",
       "Did the character in `candidate` first appear within the last ten years?", tags_true=["modern"], prior=0.35),
    _q("meta_adapted", "meta", "Has your character appeared in more than one medium?",
       "Has the character in `candidate` appeared in more than one medium?", criteria_detail="appears in more than one medium, for example both a game and an anime",
       tags_true=["multimedia"], prior=0.4),
    _q("meta_live_action", "meta", "Has your character been played by a live-action actor?",
       "Has the character in `candidate` been portrayed by a live-action actor in a film or series?",
       tags_true=["live-action"], prior=0.15),
    _q("meta_japanese_origin", "meta", "Is your character from a Japanese work?",
       "Is the source work of the character in `candidate` Japanese?", tags_true=["japanese"], prior=0.65),
    _q("meta_ensemble", "meta", "Is your character part of a large ensemble cast?",
       "Is the character in `candidate` part of a large ensemble cast?", criteria_detail="one of a large ensemble cast rather than one of only a few main characters",
       tags_true=["ensemble"], prior=0.35),

    # --- setting ---
    _q("setting_fantasy", "setting", "Is your character's world a fantasy world?",
       "Is the character in `candidate` set in a fantasy world with magic, swords or mythical creatures?",
       tags_true=["fantasy"], prior=0.35),
    _q("setting_scifi", "setting", "Is your character's world science fiction?",
       "Is the character in `candidate` in a science-fiction world?", criteria_detail="a world of spaceships, cyberpunk technology or a far future",
       tags_true=["scifi", "cyberpunk", "mecha"], prior=0.2),
    _q("setting_modern", "setting", "Is your character's story set in the modern real world?",
       "Is the character in `candidate` set in the modern-day real world?", tags_true=["modern-setting"], prior=0.3),
    _q("setting_school", "setting", "Is your character's story set mainly at a school?",
       "Is the story of the character in `candidate` set mainly in a school or academy?",
       tags_true=["school"], prior=0.2),
    _q("setting_apocalypse", "setting", "Is your character's world post-apocalyptic or at war?",
       "Is the character in `candidate` set in a post-apocalyptic world or an ongoing war?",
       tags_true=["apocalypse", "war"], prior=0.2),
]

QUESTIONS_BY_ID: dict[str, dict[str, Any]] = {q["id"]: q for q in QUESTION_BANK}
CATEGORIES: tuple[str, ...] = tuple(dict.fromkeys(q["category"] for q in QUESTION_BANK))


def make_dynamic(slug: str, phrase: str) -> dict[str, Any]:
    """Build a question from a trait phrase mined out of search snippets."""
    q = _q(
        f"dyn_{slug}",
        "mined",
        f'Is your character associated with "{phrase}"?',
        f'Is the character in `candidate` associated with "{phrase}"?',
        tags_true=[slug],
        prior=0.3,
    )
    q["dynamic"] = True
    return q
