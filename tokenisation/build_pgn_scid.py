#!/usr/bin/env python3
import time
import os
from pyscid import Database
from dotenv import load_dotenv
load_dotenv()

# Path to your test database

db_path = os.environ.get("SCID_DATABASE_PATH")
start = time.time()
db = Database.open(db_path)
count = 0
total = db.num_games
for game in db.search():
    white = game.white.replace(",", "").replace(" ", "_")
    black = game.black.replace(",", "").replace(" ", "_")
    date = game.date_string.replace(".", "")  # YYYYMMDD
    result = str(game.result).replace("/", "-").replace("-", "")  # 1-0 -> 10
    
    name = f"{white}_vs_{black}_{date}_{result}.pgn"
    with open(f"data/pgn/unsorted/{name}",'w') as pgn:
        pgn.write(game.to_pgn())
        count += 1
        print(f"{count} / {total}")
elapsed = time.perf_counter() - start
print(f"Completed in {elapsed:.2}s")
db.close()
