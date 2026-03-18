GAMES_PER_FILE = 5_000_000
MAX_GAMES = 100_000_000

# Not streaming whilst tokenizing, getting data first
from datasets import load_dataset
import argparse

#Input args
parser = argparse.ArgumentParser(description="Config for filtering Lichess games from hugging face")
parser.add_argument("--min_elo", required=False, type=int,default=0)
parser.add_argument("--output", required=True, type=str)
args = parser.parse_args()
min_elo = args.min_elo
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
for game in dataset:
    #Filter minimum elo
    white_elo = game["WhiteElo"]
    black_elo = game["BlackElo"]
    if white_elo is None or black_elo is None:
        continue
    if black_elo < min_elo or white_elo < min_elo:
        continue 
    f.write(game["movetext"] + "\n\n")
    count += 1
    #Log progressevery 1000 games
    if (count > MAX_GAMES):
        break 
    if (count % 1000 == 0):
        print(f"Games: {count} / {MAX_GAMES}")
    if count % GAMES_PER_FILE == 0:
        f.close()
        file_idx += 1
        f = open(f"{output_dir}/{file_idx:03d}.pgn", "w")


f.close()
