#!/usr/bin/env python3
import time
import os
import re
from typing import Optional, Union
from pyscid import Database
from dotenv import load_dotenv

load_dotenv()

MAX_GAMES = 15_000_000_000

FAST_EVENT_KEYWORDS = {
    "blitz",
    "rapid",
    "rapidplay",
    "bullet",
    "hyperbullet",
    "ultrabullet",
    "armageddon",
    "lightning",
    "speedchess",
    "speed chess",
    "blindfold",
    "chess960",
    "fischer random",
    "freestyle",
    "esports",
    "online",
}

NON_CLASSICAL_KEYWORDS = {
    "corr",
    "correspondence",
    "iccf",
    "iecg",
    "ficgs",
    "email",
    "thematic",
    "no_engines",
    "freestyle",
    "fischer random",
    "chess960",
    "blindfold",
    "skittles",
}

CLASSICAL_EVENT_HINTS = {
    "world championship",
    "world ch",
    "wch",
    "world cup",
    "candidates",
    "grand prix",
    "olympiad",
    "grand swiss",
    "invitational",
    "interzonal",
    "zonal",
    "masters",
    "memorial",
    "festival",
    "cup",
    "tata",
    "open",
    "championship",
    "classical",
}

CH_ABBREV_PATTERN = re.compile(r"\bch\b", re.IGNORECASE)


def is_likely_classical_event(event_name: str) -> bool:
    if not event_name or event_name == "?":
        return False

    event = event_name.lower()

    if any(keyword in event for keyword in NON_CLASSICAL_KEYWORDS):
        return False

    if any(keyword in event for keyword in FAST_EVENT_KEYWORDS):
        return False

    if any(hint in event for hint in CLASSICAL_EVENT_HINTS):
        return True

    if CH_ABBREV_PATTERN.search(event):
        return True

    return False


def elo_value(value: Optional[Union[int, str]]) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# Path to your database
db_path = os.environ.get("SCID_DATABASE_PATH")
if not db_path:
    raise RuntimeError("SCID_DATABASE_PATH is not set")

start = time.time()
db = Database.open(db_path)
count = 0
skipped_non_classical = 0
file_idx = 0
total = db.num_games
f = open(f"data/pgn/classical_GM_OTB{file_idx:03d}.pgn", "w")
for game in db.search(min_elo=2500):
    if elo_value(game.black_elo) < 2200 or elo_value(game.white_elo) < 2200:
        continue
    if not is_likely_classical_event(game.event):
        skipped_non_classical += 1
        continue
    f.write(game.to_pgn() + "\n\n")
    count += 1
    if (count % 1000 == 0):
        print(f"{count} games recorded.")
    if count % MAX_GAMES == 0:
        f.close()
        file_idx += 1
        f = open(f"data/pgn/classical_GM_OTB{file_idx:03d}.pgn", "w")
f.close()
print(f"Kept={count} skipped_non_classical={skipped_non_classical}")
db.close()
