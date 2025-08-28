import pandas as pd
import argparse
from pysgf import SGF
import os
import glob
import time
import tqdm
import random

def move_to_place(move,size):
    axis = 'abcdefghijklmnopqrstu'
    X_axis = 'ABCDEFGHJKLMNOPQRST'
    selectedPlace = 'u'
    if move == None:
        selectedPlace = 'pass'
    else:
        x = move[0]
        y = move[1]
        xaxis = axis.index(x)
        yaxis = size-axis.index(y)
        selectedPlace = str(X_axis[xaxis])+str(yaxis)
    return selectedPlace

def get_moves(node,moves):
    # get move history from node
    move = node.properties.get('B') or node.properties.get('W')
    if move:
        player = 'B' if 'B' in node.properties else 'W'
        moves.append([player,move_to_place(move[0],19)])
    for child in node.children:
        get_moves(child,moves)

def read_check(file):
    encodings = ['utf-8','ascii',None]
    # file encoding check
    read_successful = False
    for e in encodings:
        try:
            with open(file, 'r', encoding=e) as f:
                sgf_content = f.read()
            success_encoding = e
        except UnicodeDecodeError:
            continue
        else:
            read_successful = True
            break
    if read_successful == False:
        return read_successful

    # sgf encoding check
    sgf_successful = False
    for e in encodings:
        try:
            root = SGF.parse_file(file,encoding=e)
            success_sgf = e
        except:
            continue
        else:
            sgf_successful = True
            break
    if sgf_successful == False:
        return sgf_successful

    return[sgf_content,success_sgf]

def process_file(file):
    try:
        is_read = read_check(file)
        if is_read == False:
            return 
        
        e = is_read[1]
        sgf_content = is_read[0]

        root = SGF.parse_file(file,encoding=e)
        sgf_root = SGF.parse(sgf_content)

        # exclude handicap games
        if 'AB' in sgf_content:
            return
        
        idx_PB = sgf_content.find('PB[')
        idx_PW = sgf_content.find(']PW[')
        idx_BR = sgf_content.find(']BR[')

        PB = sgf_content[idx_PB+len('PB['):idx_PW]
        PW = sgf_content[idx_PW+len(']PW['):idx_BR]
        PB = PB.replace('[','(').replace(']',')')
        PW = PW.replace('[','(').replace(']',')')
    
        BR = root.get_property("BR")
        WR = root.get_property("WR")
        RE = root.get_property("RE")

        # exclude games where the ranks of the players differ
        if BR != WR:
            if BR in ['1k','2k']:
                if WR not in ['1k','2k']:
                    return
            elif BR in ['3k','4k','5k']:
                if WR not in ['3k','4k','5k']:
                    return
            else:
                return
            
        # exclude games without player names
        if (PB == '') or (PW == ''):
            return
        
        # exclude games with abnormal results
        if ('+R' not in RE) and (RE != 'draw'):
            score = float(RE[2:])
            if score < 0.5:
                return
            
        move_info = []
        get_moves(sgf_root,move_info)

        # exclude games in which passes were made
        turn_player_list = [move[0] for move in move_info]
        isin_pass = False
        for i in range(len(turn_player_list)-1):
            if turn_player_list[i] == turn_player_list[i+1]:
                isin_pass = True
                break
        if isin_pass:
            return
        if True in ['pass' in move[1] for move in move_info]:
            return 
        
        # exclude games with fewer than 50 moves
        if len(move_info)<=50:
            return
        data = [file,PB,BR,PW,WR]

        return data
    except Exception as e:
        return

def select_game_records(rank_list,target):
    results = []
    start = time.time()
    file_list = []
    base_file_list = []
    for rank in rank_list:
        files = glob.glob(f'{target}{rank}*/*.sgf')
        base_files = [os.path.basename(file) for file in files]
        file_list += files
        base_file_list += base_files
    print('total files:',len(base_file_list))
    
    # select game records
    for i in tqdm.tqdm(range(len(base_file_list))):
        if i % 10000 == 0:
            print(rank_list,i,len(results))
            
        target_file = file_list[i]
        result = process_file(target_file)
        if result:
            results.append(result)
    end = time.time()
    print('finished',rank_list,':',format(end-start,'.1f'),'seconds')
    return results

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
            datasize = 100
        elif type == 'testing':
            datasize = 900
        elif type == 'training':
            datasize = 5000
        sample = game_info.sample(n=datasize)

    remaining_games = remaining_games.drop(index=sample.index)
    sample.reset_index(drop=True).to_csv(
        os.path.join(path,f'{label}_info.csv'),index=False
    )
    target_files = sample['file'].unique().tolist()
    output_path = os.path.join(path,f'{label}.sgf')
    texts = []
    for file in target_files:
        with open(file,'r') as f:
            text = f.read()
        texts.append(text)
    with open(output_path,'w') as f:
        f.write('\n'.join(texts))
    return remaining_games
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, required=True, help='Target directory to analyze')
    args = parser.parse_args()

    dir_path = os.getcwd()
        
    ranks = [
        ['9d'],['8d'],['7d'],['6d'],['5d'],
        ['4d'],['3d'],['2d'],['1d'],
        ['1k','2k'],['3k','4k','5k']
    ]
    labels = ['9d','8d','7d','6d','5d','4d',
              '3d','2d','1d','1-2k','3-5k']
    random.seed(256)

    for idx,rank_list in enumerate(ranks):
        game_records = select_game_records(rank_list,args.target)

        game_info = pd.DataFrame(
            game_records,
            columns=[
                'file','PB','BR','PW','WR'
            ]
        )

        # extract data 
        ## teststing by player 
        if not os.path.isdir(os.path.join(dir_path,'testing_sgf_go_by_player')):
            os.mkdir(os.path.join(dir_path,'testing_sgf_go_by_player'))
        test_path = os.path.join(dir_path,'testing_sgf_go_by_player')
        game_info = extract_data('testing_by_player',game_info,test_path,labels[idx])
        
        ## candidating
        if not os.path.isdir(os.path.join(dir_path,'candidating_sgf_go')):
            os.mkdir(os.path.join(dir_path,'candidating_sgf_go'))
        cand_path = os.path.join(dir_path,'candidating_sgf_go')
        game_info = extract_data('candidating',game_info,cand_path,labels[idx])
        
        ## training
        if not os.path.isdir(os.path.join(dir_path,'training_sgf_go')):
            os.mkdir(os.path.join(dir_path,'training_sgf_go'))
        train_path = os.path.join(dir_path,'training_sgf_go')
        game_info = extract_data('training',game_info,train_path,labels[idx])
        
        ## testing
        if not os.path.isdir(os.path.join(dir_path,'testing_sgf_go')):
            os.mkdir(os.path.join(dir_path,'testing_sgf_go'))
        test_path = os.path.join(dir_path,'testing_sgf_go')
        extract_data('testing',game_info,test_path,labels[idx])

if __name__ == '__main__':
    main()