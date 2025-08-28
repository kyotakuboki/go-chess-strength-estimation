from pysgf import SGF
import numpy as np
import os
import pandas as pd
import argparse
import tqdm
import gzip
import glob
import json
import re

def move_to_idx(move):
    if move == 'pass':
        return -1
    else:
        x = ord(move[0]) - ord('A') - (1 if move[0] > 'I' else 0) 
        y = 19 - int(move[1:])
        return 19 * y + x
    
def process_file(jsonl_file):
    try:
        with gzip.open(jsonl_file,'r') as f:
            lines = [json.loads(line) for line in f.readlines()]
    except Exception as e:
        return 
    
    file = lines[0]['sgf_path']
    
    history_info = lines[0]['history_moves']
    history_move = [move[1] for move in lines[0]['history_moves']]

    priors = [
        'preaz_9d','preaz_7d',
        'preaz_5d','preaz_3d','preaz_1d',
        'preaz_2k','preaz_4k','preaz_6k',
        'preaz_8k','preaz_10k',
        'prior'
    ]
    base_info_l = [
        "".join(history_move),file
    ]
    turn_info_l = []
    for turn in range(len(history_info)):
        try:
            line = lines[turn+1]
            played_move = history_info[turn][1]
            
            moveInfos = line['moveInfos']
            playedInfos = [infos for infos in moveInfos if infos['move']==played_move][0]
            
            root_scorelead = line['rootInfo'][f'scoreLead']
            played_scorelead = playedInfos['scoreLead']      
            played_scloss = root_scorelead - played_scorelead   
            
            played_priors = [playedInfos[prior] for prior in priors]      
            data = [
                turn,played_move,len(history_info),
                root_scorelead
            ]+played_priors+[played_scorelead,played_scloss]
            turn_info_l.append(base_info_l+data)
        except Exception as e:
            print(e)
            return 

    return turn_info_l

def get_moves(node,moves):
    move = node.properties.get('B') or node.properties.get('W')
    if move:
        player = 'B' if 'B' in node.properties else 'W'
        moves.append(player)
    for child in node.children:
        get_moves(child,moves)

def extract_moves_from_sgf(sgf):
    moves = []
    if len(sgf.children) > 0:
        node = sgf.children[0]
        while True:
            if node.move is not None and node.move.player is not None:
                moves.append([node.move.player,node.move.gtp()])
            if len(node.children) == 0:
                break
            node = node.children[0]
    return moves

def parse_sgf(sgf_list,csv_list,target):
    game_info = []
    for i in range(len(sgf_list)):
        file_path = sgf_list[i]
        csv_path = csv_list[i]
        
        with open(os.path.join(target,file_path),'r',encoding='utf-8') as file:
            sgf_data = file.readlines()
            
        with open(os.path.join(target,csv_path),'r',encoding='utf-8') as file:
            csv_data = file.readlines()
            
        for game,csv in zip(sgf_data,csv_data):
            sgf_root = SGF.parse(game)
            move_info = []
            get_moves(sgf_root,move_info)
            history_moves = extract_moves_from_sgf(sgf_root)
            history_moves = [move[1] for move in history_moves]
            n_moves = len(move_info)
            strength_list = csv.split(',')[:-1]
            
            pb_match = re.search(r"PB\[(.*?)\]", game)  
            pw_match = re.search(r"PW\[(.*?)\]", game)  
            br_match = re.search(r"BR\[(.*?)\]", game)  
            wr_match = re.search(r"WR\[(.*?)\]", game)        
            re_match = re.search(r"RE\[(.*?)\]", game)  
            
            pb = pb_match.group(1) if pb_match else "Unknown"
            pw = pw_match.group(1) if pw_match else "Unknown"
            br = br_match.group(1) if br_match else "Unknown"
            wr = wr_match.group(1) if wr_match else "Unknown"
            res = re_match.group(1) if re_match else "Unknown"
            
            # win = 'u'
            # if 'B' in res:
            #     win = pb
            # elif 'W' in res:
            #     win = pw
            # elif 'draw' in res:
            #     win = 'draw'
            
            for turn in range(n_moves):
                game_info.append({
                    'history_moves':"".join(history_moves),
                    'PB':pb,'PW':pw,'BR':br,'WR':wr,'turn':turn,
                    'Target':move_info[turn],'s-score':strength_list[turn],
                    'n_moves':n_moves
                })
    return pd.DataFrame(game_info)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, required=True)
    args = parser.parse_args()

    group_list = [
        '3-5k','1-2k','1d','2d','3d','4d','5d','6d','7d','8d','9d'
    ]
    sgf_files = [f'{group}.sgf' for group in group_list]
    csv_files = [f'{group}.csv' for group in group_list]
    jsonl_files = [f'{group}_analyzed' for group in group_list]
    
    str_df = parse_sgf(sgf_files,csv_files,args.target)
    
    id = 0
    for group,jsonl_file in zip(group_list,jsonl_files):
        print(group)
        info_l = []
        for file_id in tqdm.tqdm(range(len(glob.glob(f'{os.path.join(args.target,jsonl_file)}/*.jsonl.gz')))):
            file = f'{os.path.join(args.target,jsonl_file)}/{group}_{file_id}_analyzed_{file_id}.jsonl.gz'
            result = process_file(file)
            if result:
                info_l += [a+[id] for a in result]
            id += 1
    
        df = pd.DataFrame(
            info_l,
            columns=[
                'history_moves','file',
                'turn','move','n_moves',
                'root_value',
                'prior_9d','prior_7d','prior_5d',
                'prior_3d','prior_1d','prior_2k','prior_4k',
                'prior_6k','prior_8k','prior_10k',
                'prior_kt',
                'value',
                'loss',
                'id',
            ]
        )
        turn_df = pd.merge(df,str_df,on=['turn','n_moves','history_moves'],how='inner').drop(columns=['history_moves'])
        turn_df['group'] = [
        0 if a in ['3k','4k','5k'] else
        1 if a in ['1k','2k'] else
        2 if a == '1d' else
        3 if a == '2d' else
        4 if a == '3d' else
        5 if a == '4d' else
        6 if a == '5d' else
        7 if a == '6d' else
        8 if a == '7d' else
        9 if a == '8d' else
        10 if a == '9d' else
        np.nan for a in turn_df['BR']
        ]
        print(turn_df['group'].value_counts())
        print(turn_df['id'].nunique())
        for i in turn_df['group'].unique():
            target_df = turn_df[turn_df['group']==i]
            target_df.to_csv(os.path.join(args.target,f'{group}_turn_info.csv'),index=False)
        
if __name__ == '__main__':
    main()