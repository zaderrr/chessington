GAMES_PER_FILE = 5_000_000
MAX_GAMES = 100_000_000_000

# Not streaming whilst tokenizing, getting data first
from datasets import load_dataset
import argparse

def is_good_game(g):
    if g["WhiteElo"] is None or g["BlackElo"] is None:
        return False
    if g["WhiteElo"] < min_elo or g["BlackElo"] < min_elo:
        return False
    if abs(g["WhiteElo"] - g["BlackElo"]) > max_diff:
        return False
    if g["Termination"] != "Normal":
        return False
    try:
        base = int(g["TimeControl"].split("+")[0])
        if base < 600:
            return False
    except:
        return False
    return True
#Input args
parser = argparse.ArgumentParser(description="Config for filtering Lichess games from hugging face")
parser.add_argument("--min_elo", required=False, type=int,default=0)
parser.add_argument("--max_diff", required=False, type=int,default=2000)
parser.add_argument("--output", required=True, type=str)
args = parser.parse_args()
min_elo = args.min_elo
max_diff = args.max_diff
output_dir= args.output
#Load dataset
dataset =  load_dataset(
"Lichess/standard-chess-games",
split="train",
streaming=True,
) 
file_idx = 0
count = 0


f = open(f"{output_dir}/{file_idx:03d}.pgn", "w")

dataset = dataset.filter(is_good_game)

for game in dataset:
    f.write(game["movetext"] + "\n\n")
    count += 1
    #Log progressevery 1000 games
    if (count % 1000 == 0):
        print(f"Games: {count} / {MAX_GAMES}")
    if count % GAMES_PER_FILE == 0:
        f.close()
        file_idx += 1
        f = open(f"{output_dir}/{file_idx:03d}.pgn", "w")
    if (count > MAX_GAMES):
        break 
f.close()
