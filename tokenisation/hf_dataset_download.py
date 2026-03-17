GAMES_PER_FILE = 5_000_000
# Not streaming whilst tokenizing, getting data first
from datasets import load_dataset
# Lichess/standard-chess-games

year = "xxxx"
month = "xx"

dataset =  load_dataset(
"Lichess/standard-chess-games",
data_dir=f"data/year={year}/month={month}",
split="train",
streaming=True,
) 


file_idx = 0
count = 0
f = open(f"data/pgn/lichess_{year}_{month}_{file_idx:03d}.pgn", "w")
for game in dataset:
    f.write(game["movetext"] + "\n\n")
    count += 1
    print (f"{count} games stored." )
    if count % GAMES_PER_FILE == 0:
        f.close()
        file_idx += 1
        f = open(f"lichess_{year}_{month}_{file_idx:03d}.pgn", "w")

f.close()
