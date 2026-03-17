#!/usr/bin/env python3
import time
import os
from pyscid import Database
from dotenv import load_dotenv
load_dotenv()

MAX_GAMES = 15_000_000_000

# Path to your database
db_path = os.environ.get("SCID_DATABASE_PATH")
start = time.time()
db = Database.open(db_path)
count = 0
file_idx = 0
total = db.num_games
f = open(f"data/pgn/unsortedGIGABASE_{file_idx:03d}.pgn", "w");
for game in db.search():
        f.write(game.to_pgn())
        count += 1
        print(f"{count} / {total}")
        if count % MAX_GAMES == 0:
            f.close()
            file_idx += 1
            f = open(f"data/pgn/unsortedGIGABASE_{file_idx:03d}.pgn", "w");
elapsed = time.perf_counter() - start
print(f"Completed in {elapsed:.2}s")
db.close()
