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

def value2cp(v):
    alpha = 1e-10
    winrate = (v+1)/2
    if winrate == 0:
        P = alpha
    elif winrate == 1:
        P = winrate - alpha
    else:
        P = winrate

    return np.log(P/(1-P))

def process_file(jsonl_file):
    try:
        with gzip.open(jsonl_file,'r') as f:
            lines = [json.loads(line) for line in f.readlines()]
    except Exception as e:
        return 
    
    file = lines[0]['Site']
    history_moves = lines[0]['history_moves']
    PB = lines[0]['PB']
    PW = lines[0]['PW']
    BR = int(lines[0]['BR'])
    WR = int(lines[0]['WR'])
    # Event = lines[0]['Event']
    # Result = lines[0]['Result']
    
    policies = [
        'maia-1100','maia-1200','maia-1300',
        'maia-1400','maia-1500','maia-1600',
        'maia-1700','maia-1800','maia-1900',
        'lc0'
    ]
    base_info_l = [
        PB,PW,BR,WR,file,
    ]
    turn_info_l = []
    for turn in range(len(history_moves)):
        line = lines[turn+1]
        played_move = history_moves[turn]
        
        moveInfos = line['moveInfos']
        try:
            playedInfos = [infos for infos in moveInfos if infos['move']==played_move][0]
        except Exception as e: 
            if played_move == 'e8g8':
                playedInfos = [infos for infos in moveInfos if infos['move']=='e8h8'][0]
            elif played_move == 'e1g1':
                playedInfos = [infos for infos in moveInfos if infos['move']=='e1h1'][0]
            elif played_move == 'e1c1':
                playedInfos = [infos for infos in moveInfos if infos['move']=='e1a1'][0]
            elif played_move == 'e8c8':
                playedInfos = [infos for infos in moveInfos if infos['move']=='e8a8'][0]
            else:
                print(played_move,[infos['move'] for infos in moveInfos])
                break
        
        root_value = line['rootInfo']['lc0_value']
        root_cp = value2cp(root_value)
        played_value = playedInfos['lc0_value']      
        played_cp = value2cp(played_value)
        played_cploss = root_cp - played_cp     
        
        played_priors = []      
        for policy in policies:
            candidates = []
            move_and_prior_list = [{'move':infos['move'],'prior':infos[f'{policy}_prior']} for infos in moveInfos]
            move_and_prior_list.sort(key=lambda x:x['prior'], reverse=True)
            
            for i in range(min(len(move_and_prior_list),20)):
                move = move_and_prior_list[i]['move']
                candidates.append(move)
            
            if played_move not in candidates:
                if (played_move not in  ['e8g8','e1g1','e1c1','e8c8']):
                    candidates.append(played_move)
                    
            played_prior = playedInfos[f'{policy}_prior']
            played_priors.append(played_prior)
            
        data = [
            turn,played_move,len(history_moves),
            root_cp
        ]+played_priors+[played_cp,played_cploss]
        turn_info_l.append(base_info_l+data)

    return turn_info_l

def get_moves(node,moves):
    move = node.properties.get('B') or node.properties.get('W')
    if move:
        player = 'B' if 'B' in node.properties else 'W'
        moves.append(player)
    for child in node.children:
        get_moves(child,moves)

def parse_sgf(txt_list,csv_list,target):
    game_info = []
    for i in range(len(txt_list)):
        file_path = txt_list[i]
        csv_path = csv_list[i]
        
        with open(os.path.join(target,file_path),'r',encoding='utf-8') as file:
            sgf_data = file.readlines()
        
        with open(os.path.join(target,csv_path),'r',encoding='utf-8') as file:
            csv_data = file.readlines()
        
        for game,csv in zip(sgf_data,csv_data):
            sgf_root = SGF.parse(game)
            move_info = []
            get_moves(sgf_root,move_info)
            n_moves = len(move_info)
            strength_list = csv.split(',')[:-1]
            
            pb_match = re.search(r"PB\[(.*?)\]", game) 
            pw_match = re.search(r"PW\[(.*?)\]", game) 
            br_match = re.search(r"BR\[(.*?)\]", game) 
            wr_match = re.search(r"WR\[(.*?)\]", game)  

            pb = pb_match.group(1) if pb_match else "Unknown"
            pw = pw_match.group(1) if pw_match else "Unknown"
            br = br_match.group(1) if br_match else "Unknown"
            wr = wr_match.group(1) if wr_match else "Unknown"
            
            for turn in range(n_moves):
                game_info.append({
                    'PB':pb,'PW':pw,'BR':int(br),'WR':int(wr),'turn':turn,
                    'Target':move_info[turn],'s-score':strength_list[turn],
                    'n_moves':n_moves
                })
    return pd.DataFrame(game_info)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target',type=str,required=True)
    args = parser.parse_args()
    
    group_list = [
        '1000_1200','1200_1400','1400_1600','1600_1800',
        '1800_2000','2000_2200','2200_2400','2400_2600'
    ]

    txt_files = [f'{group}.txt' for group in group_list]
    csv_files = [f'{group}.csv' for group in group_list]
    if args.target == 'testing_by_player':
        jsonl_files = [f'{group}_analyzed' for group in group_list]
    else:
        jsonl_files = [f'{args.target}_analyzed']
    
    str_df = parse_sgf(txt_files,csv_files,args.target)

    info_l = []
    for idx,jsonl in enumerate(jsonl_files):
        print(jsonl)
        for file_id in tqdm.tqdm(range(len(glob.glob(f'{os.path.join(args.target,jsonl)}/*.jsonl.gz')))):
            if args.target == 'testing_by_player':
                file = f'{os.path.join(args.target,jsonl)}/{group_list[idx]}_{file_id}_analyzed.jsonl.gz'
            else:
                file = f'{os.path.join(args.target,jsonl)}/{args.target}_{file_id}_analyzed.jsonl.gz'
            result = process_file(file)
            if result:
                info_l += [a+[file_id] for a in result]
    
    df = pd.DataFrame(
        info_l,
        columns=[
            'PB','PW','BR','WR','file',
            'turn','move','n_moves',
            'root_value',
            'prior_1100','prior_1200','prior_1300',
            'prior_1400','prior_1500','prior_1600',
            'prior_1700','prior_1800','prior_1900',
            'prior_lc0',
            'value',
            'loss',
            'id'
        ]
    )
    
    turn_df = pd.merge(df,str_df,on=['PB','PW','BR','WR','turn','n_moves'],how='inner')
    turn_df['group'] = [
        0 if 1000 <= a and a < 1200 else
        1 if 1200 <= a and a < 1400 else
        2 if 1400 <= a and a < 1600 else
        3 if 1600 <= a and a < 1800 else
        4 if 1800 <= a and a < 2000 else
        5 if 2000 <= a and a < 2200 else
        6 if 2200 <= a and a < 2400 else
        7 if 2400 <= a and a < 2600 else
        np.nan for a in turn_df['BR']
    ]
    for i in turn_df['group'].unique():
        target_df = turn_df[turn_df['group']==i]
        target_df.to_csv(os.path.join(args.target,f'{group_list[i]}_turn_info.csv'),index=False)
    
if __name__ == '__main__':
    main()