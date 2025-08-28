import pandas as pd
from pysgf import SGF
import os
import argparse
import re
import time
import random
from sklearn.metrics import confusion_matrix, accuracy_score

def calculate_s_score_mean(type):
    if type == 'chess':
        rank_list = [
            '1000_1200','1200_1400','1400_1600',
            '1600_1800','1800_2000','2000_2200',
            '2200_2400','2400_2600'
        ]
    elif type == 'go':
        rank_list = [
            '3-5k','1-2k','1d','2d','3d',
            '4d','5d','6d','7d','8d','9d'
        ]  
    s_score_mean_list = []
    for rank in rank_list:
        csv_filename = os.path.join(f'{type}/candidating',f'{rank}.csv')
        
        with open(csv_filename,'r') as csv_file:
            csv_data = csv_file.readlines()
        
        s_score = 0
        len_s_score = 0
        for csv in csv_data:
            target_s_score_l = [float(s_score) for s_score in csv.split(',')[:-1]]
            s_score += sum(target_s_score_l)
            len_s_score += len(target_s_score_l)
        s_score_mean_list.append(s_score/len_s_score)
    return s_score_mean_list

def extract_game_info_chess(path):
    def get_moves(node,moves):
        move = node.properties.get('B') or node.properties.get('W')
        if move:
            player = 'B' if 'B' in node.properties else 'W'
            moves.append(player)
        for child in node.children:
            get_moves(child,moves)
            
    rank_list = [
        '1000_1200','1200_1400','1400_1600',
        '1600_1800','1800_2000','2000_2200',
        '2200_2400','2400_2600'
    ]
    game_info = []
    for i in range(len(rank_list)):
        sgf_filename = f'{rank_list[i]}.txt'
        csv_filename = f'{rank_list[i]}.csv'
        with open(os.path.join(path,csv_filename),'r') as file:
            csv_data = file.readlines()
            
        with open(os.path.join(path,sgf_filename),'r') as file:
            sgf_data = file.readlines()
            
        game_id = 0
        for game,csv in zip(sgf_data,csv_data):
            sgf_root = SGF.parse(game)
            move_info = []
            get_moves(sgf_root,move_info)
            n_moves = len(move_info)
            s_score_list = [float(s_score) for s_score in csv.split(',')[:-1]]
            
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
                    'PB':pb,'PW':pw,'BR':br,'WR':wr,'rank':i,'turn':turn,
                    's-score':s_score_list[turn],'game_id':game_id
                })
            game_id += 1
    return pd.DataFrame(game_info)

def extract_game_info_go(path):
    def get_moves(node,moves):
        move = node.properties.get('B') or node.properties.get('W')
        if move:
            player = 'B' if 'B' in node.properties else 'W'
            moves.append(player)
        for child in node.children:
            get_moves(child,moves)
            
    rank_list = [
        '3-5k','1-2k','1d','2d','3d',
        '4d','5d','6d','7d','8d','9d'
    ]
    game_info = []
    for i in range(len(rank_list)):
        sgf_filename = f'{rank_list[i]}.sgf'
        csv_filename = f'{rank_list[i]}.csv'
        with open(os.path.join(path,csv_filename),'r') as file:
            csv_data = file.readlines()
            
        with open(os.path.join(path,sgf_filename),'r') as file:
            sgf_data = file.readlines()
            
        game_id = 0
        for game,csv in zip(sgf_data,csv_data):
            sgf_root = SGF.parse(game)
            move_info = []
            get_moves(sgf_root,move_info)
            n_moves = len(move_info)
            s_score_list = [float(s_score) for s_score in csv.split(',')[:-1]]
            
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
                    'PB':pb,'PW':pw,'BR':br,'WR':wr,'rank':i,'turn':turn,
                    's-score':s_score_list[turn],'game_id':game_id
                })
            game_id += 1
    return pd.DataFrame(game_info)

def find_nearest_index(l,value):
    return min(range(len(l)),key=lambda i:abs(l[i]-value))

def predict(type,cand_by_rank,game_info,by_player=False):
    if by_player:
        repeat_times = 5
        n_list = [5,10,15]
    else:
        repeat_times = 500
        n_list = [1,5,10,15,20]
    
    if type == 'chess':
        rank_list = [
            '1000_1200','1200_1400','1400_1600',
            '1600_1800','1800_2000','2000_2200',
            '2200_2400','2400_2600'
        ]
    elif type == 'go':
        rank_list = [
            '3-5k','1-2k','1d','2d','3d',
            '4d','5d','6d','7d','8d','9d'
        ] 
    
    total_results = []
    for n_used_games in n_list:
        rank_results = []
        player_list = []
        for rank in range(len(rank_list)):
            games_data = game_info[game_info['rank']==rank]
            
            if by_player:
                id_data = games_data.drop_duplicates(subset=['game_id'])
                PBs = id_data['PB'].value_counts()
                PWs = id_data['PW'].value_counts()
                players = PBs.add(PWs,fill_value=0)
                players20 = players[players>=20].index
                player_list += players20[:100].tolist()
                for player in players20[:100]:
                    player_data = games_data[(games_data['PB']==player)|(games_data['PW']==player)]
                    player_games = player_data['game_id'].unique().tolist()
                    for _ in range(repeat_times):
                        score = 0
                        num_position = 0
                        target_games = random.sample(player_games[:20],n_used_games)
                        for target_game in target_games:
                            target_data = player_data[player_data['game_id']==target_game]
                            pos = 'u'
                            if player == target_data['PB'].values[0]:
                                pos = 0
                            elif player == target_data['PW'].values[0]:
                                pos = 1
                            temp = 0
                            lower_bound = 0
                            upper_bound = 1000
                            
                            for turn in range(len(target_data)):
                                if (temp >= lower_bound)&(turn % 2 == pos):
                                    turn_info = target_data[target_data['turn']==turn]
                                    score += turn_info['s-score'].values[0]
                                    num_position += 1
                                temp += 1
                                if temp >= upper_bound:
                                    break
                        average_score = score / num_position
                        predicted_rank = find_nearest_index(cand_by_rank,average_score)
                        rank_results.append([predicted_rank,average_score,rank])
            else:
                games_id = games_data['game_id'].unique().tolist()
                
                for _ in range(repeat_times):
                    score = 0
                    num_position = 0
                    target_games = random.sample(games_id,n_used_games)
                    for target_game in target_games:
                        target_data = games_data[games_data['game_id']==target_game]
                        
                        pos = random.randint(0,1)               
                        temp = 0
                        lower_bound = 0
                        upper_bound = 1000
                        
                        for turn in range(len(target_data)):
                            if (temp >= lower_bound)&(turn % 2 == pos):
                                turn_info = target_data[target_data['turn']==turn]
                                score += turn_info['s-score'].values[0]
                                num_position += 1
                            temp += 1
                            if temp >= upper_bound:
                                break
                    average_score = score / num_position
                    predicted_rank = find_nearest_index(cand_by_rank,average_score)
                    rank_results.append([predicted_rank,average_score,rank])
        results_info = pd.DataFrame(
            rank_results,
            columns=['y_pred',' s-score','rank'],
        ).rename_axis(f'n={n_used_games}')
        total_results.append(results_info)
        t = time.time()
        print(f'finished n={n_used_games}')
    return total_results

def accuracy1_score(conf_matrix):
    conf_sum = 0
    c = 0
    for i in range(len(conf_matrix)):
        row_sum = sum(conf_matrix[i])
        i_minus_1 = conf_matrix[i][i-1] if i-1>=0 else 0
        i_exact = conf_matrix[i][i]
        i_plus_1 = conf_matrix[i][i+1] if i+1<len(conf_matrix) else 0
        conf_sum += row_sum
        c += i_minus_1 + i_exact + i_plus_1
    return c/conf_sum

def save_results(results,n_list,target,by_player=False):
    results_l = []
    for result in results:
        accuracy_l = []
        accuracy1_l = []
        y_pred = result['y_pred']
        y = result['rank']
        accuracy = accuracy_score(y,y_pred)
        conf_matrix = confusion_matrix(y,y_pred)
        accuracy1 = accuracy1_score(conf_matrix)
        accuracy_l.append(accuracy)
        accuracy1_l.append(accuracy1)
        display_score = f'{format(accuracy*100,".1f")} ({format(accuracy1*100,".1f")})'
        results_l.append(display_score)
    df = pd.DataFrame(
        results_l,columns=['accuracy (accuracy1)'],
        index=[f'n={n}' for n in n_list]
    )
    root = '_by_player' if by_player else ''
    df.to_csv(f'results/{target}/accuracy_info_chen_method{root}.csv')
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target',type=str,required=True)
    args = parser.parse_args()

    s_score_by_cand = calculate_s_score_mean(args.target)
    print(f'calculated s-score mean from candidate')
    print([format(s_score,'.3f') for s_score in s_score_by_cand])
    
    if args.target == 'chess':
        game_info = extract_game_info_chess(f'{args.target}/testing')
        game_info2 = extract_game_info_chess(f'{args.target}/testing_by_player')
    elif args.target == 'go':
        game_info = extract_game_info_go(f'{args.target}/testing')
        game_info2 = extract_game_info_go(f'{args.target}/testing_by_player')
    t = time.time()
    print(f'extracted game info from test')
    
    # main results
    print('main results')
    results = predict(args.target,s_score_by_cand,game_info)
    t = time.time()
    print(f'predicted')
    
    n_list = [1,5,10,15,20]
    for idx,result in enumerate(results):
        result.to_csv(f'results/{args.target}/chen_method_{n_list[idx]}.csv')
    t = time.time()
    
    save_results(results,n_list,args.target)
    print(f'save results')

    # by-player results
    print('by-player results')
    results = predict(args.target,s_score_by_cand,game_info2,by_player=True)
    t = time.time()
    print(f'predicted')
    
    n_list = [5,10,15]
    for idx,result in enumerate(results):
        result.to_csv(f'results/{args.target}/chen_method_{n_list[idx]}_by_player.csv')
    t = time.time()
    
    save_results(results,n_list,args.target,by_player=True)
    print(f'save results')

if __name__ == '__main__':
    main()