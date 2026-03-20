import re
from berserk import TokenSession,Client, exceptions
from dotenv import load_dotenv
import argparse
import os 
import csv
import time
import json
MAX_PUZZLES = 10000
load_dotenv(".env")
api_token = os.getenv("lichess_key")

parser = argparse.ArgumentParser(description="Config for downloading full games found in the lichess puzzle database")
parser.add_argument("--source", required=True, type=str,default="")
parser.add_argument("--output", required=False, type=str,default="pgn_puzzles.json")
args = parser.parse_args()
sourceDir = args.source
outputDir = args.output

source = open(sourceDir, "r")
puzzle_reader = csv.reader(source, delimiter=',')
output = open(outputDir, "w")
game_list = []
output.write("[")   
session = TokenSession(api_token)
client = Client(session=session)

count = 0

# Source file headers = PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

def get_games(game_ids):
    try:
        game_moves = []
        games = client.games.export_multi(*game_ids, as_pgn=True)
        for game in games:
            #Remove PGN metadata
            moves = game.split("\n\n")[-1]
            game_moves.append(moves)
        return game_moves
    except Exception as e :
        print(f"Error: {e}")

def get_id_from_url(url):
    match = re.search(r'lichess\.org/([^/#]+)', url)
    if match:
        return match.group(1)
# Skip headers
next(puzzle_reader)

for row in puzzle_reader:
    if (count == MAX_PUZZLES):
        break
    
    if (count % 300 == 0):
        game_moves = get_games(g["ID"] for g in game_list)
        for x in range(0, len(game_list) -1):
            game_list[x]["PGN"] = game_moves[x] 
        for game in game_list:
            output.write(json.dumps(game) + ",\n")
        game_list.clear()

    starting_pos = row[1]
    puzzle_rating = row[3]
    themes = row[7].split(' ')
    game_id = get_id_from_url(row[8])
    moves = row[2]
    opening_tags = row[9].split(' ')
    game = {"Starting position": starting_pos, "Moves": moves ,"Rating": puzzle_rating, "Themes": themes, "ID": game_id, "Opening Tags": opening_tags, "PGN": ""}
    game_list.append(game)
    count += 1
    if (count % 1000 == 0):
        print(f"{count} puzzles processed")
output.write("]")
output.close()
source.close()
