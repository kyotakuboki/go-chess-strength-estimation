import os
import chess
import chess.pgn
import pandas as pd
import argparse
import bisect
import glob
import re
import tqdm
import io

rating_list = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
rank_list = list(range(len(rating_list)-1))

def read_pgn_games(pgn_path):
    with open(pgn_path,'r',encoding='utf-8') as pgn:
        content = pgn.read()
    
    games = re.split(r'\n\s*\n(?=\[Event )', content)
    return games

def get_rank(rating):
    idx = bisect.bisect_right(rating_list, rating) - 1
    return rank_list[idx] if 0 <= idx < len(rank_list) else None

def process_game(game_text):
    try:
        game_io = io.StringIO(game_text)
        game = chess.pgn.read_game(game_io)
        
        if game is None:
            return
        
        White, Black = game.headers.get("White", "?"), game.headers.get("Black", "?")
        # exclude games without player names
        if White == "?" or Black == "?":
            return

        try:
            WhiteElo, BlackElo = int(game.headers.get("WhiteElo", -1)), int(game.headers.get("BlackElo", -1))
        except ValueError:
            return 

        White_rank, Black_rank = get_rank(WhiteElo), get_rank(BlackElo)
        # exclude games where the ranks of the players differ and games not in the target ranks
        if White_rank is None or Black_rank is None or White_rank != Black_rank:
            return 
        
        # exclude games played in formats other than Blitz
        if "Blitz" not in game.headers.get("Event", ""):
            return

        # exclude games with fewer than 20 moves
        if len(list(game.mainline_moves())) <= 20:
            return
        
        Site = game.headers.get("Site","")
        data = [Site,Black,Black_rank,White,White_rank,game]
        
        return data
    except Exception as e:
        return

def select_game_records(target):
    games = read_pgn_games(target)
    results = []
    for game in tqdm(games):
        result = process_game(game)
        if result:
            results.append(result)
    return results

def write_games_to_gpn(games,output_path):
    with open(output_path,'w',coding='utf-8') as f:
        for game in games:
            exporter = chess.pgn.StringExpoter(headers=True,variations=True,comments=True)
            pgn_string = game.accept(exporter)
            f.write(pgn_string)
            f.write('\n\n')

def extract_data(type,game_info,path,label):
    remaining_games = game_info.copy()
        
    if type == 'testing_by_player':
        # extract players with 20 or more games
        player20 = game_info['PB'].value_counts()
        player20 = player20.add(game_info['PW'].value_counts(),fill_value=0)
        player20 = player20[player20>=20].index
        for player in player20[:100]:
            df = game_info[(game_info['PB']==player)|(game_info['PW']==player)]
            sample = pd.concat([df.sample(n=20),sample])
    else:
        if type == 'candidating':
            datasize = 120
        elif type == 'testing':
            datasize = 1200
        elif type == 'training':
            datasize = 13000
        sample = game_info.sample(n=datasize)
    remaining_games = remaining_games.drop(index=sample.index)
    sample.reset_index(drop=True).to_csv(
        os.path.join(path,f'{label}_info.csv'),index=False
    )
    target_files = sample['file'].unique().tolist()
    output_path = os.path.join(path,f'{label}.pgn')
    write_games_to_gpn(target_files,output_path)
    return remaining_games
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, required=True, help='PGN file to process')
    args = parser.parse_args()
    
    dir_path = os.getcwd()
    
    labels = ['1000_1200','1200_1400','1400_1600','1600_1800',
              '1800_2000','2000_2200','2200_2400','2400_2600']
    
    game_records = select_game_records(args.target)
    game_info = pd.DataFrame(
        game_records,
        columns=['Site','PB','BR','PW','WR','game']
    )
    
    for idx,rank in enumerate(rank_list):
        df = game_info[game_info['BR']==rank]
        
        # extract data
        ## testing by player
        if not os.path.isdir(os.path.join(dir_path,'testing_by_player')):
            os.mkdir(os.path.join(dir_path,'testing_by_player'))
        test_path = os.path.join(dir_path,'testing_by_player')
        df = extract_data('testing_by_player',df,test_path,labels[idx])
        
        ## candidating
        if not os.path.isdir(os.path.join(dir_path,'candidating')):
            os.mkdir(os.path.join(dir_path,'candidating'))
        cand_path = os.path.join(dir_path,'candidating')
        df = extract_data('candidating',df,cand_path,labels[idx])
        
        ## training
        if not os.path.isdir(os.path.join(dir_path,'training')):
            os.mkdir(os.path.join(dir_path,'training'))
        train_path = os.path.join(dir_path,'training')
        df = extract_data('training',df,train_path,labels[idx])
        
        ## testing
        if not os.path.isdir(os.path.join(dir_path,'testing')):
            os.mkdir(os.path.join(dir_path,'testing'))
        test_path = os.path.join(dir_path,'testing')
        extract_data('testing',df,test_path,labels[idx])

    
if __name__ == '__main__':
    main()