import argparse
import time
import json
import os

from lczero.backends import Weights, Backend, GameState
import chess.pgn

maia_setting_path = 'The path to your Leela Chess Zero script directory'

weight_files = [
    'maia-1100.pb.gz',
    'maia-1200.pb.gz',
    'maia-1300.pb.gz',
    'maia-1400.pb.gz',
    'maia-1500.pb.gz',
    'maia-1600.pb.gz',
    'maia-1700.pb.gz',
    'maia-1800.pb.gz',
    'maia-1900.pb.gz',
    '195b450999e874d07aea2c09fd0db5eff9d4441ec1ad5a60a140fe8ea94c4f3a'
]

model_labels = [
    'maia-1100','maia-1200','maia-1300','maia-1400',
    'maia-1500','maia-1600','maia-1700','maia-1800',
    'maia-1900','lc0'
]

def save_result(filepath,game_info,i):
    basename = os.path.basename(filepath)
    savedir_path = os.path.join(os.path.dirname(filepath),f'{basename[:-len(".sgf")]}_analyzed')
    if not os.path.isdir(savedir_path):
        os.mkdir(savedir_path)
    
    jsonl_path = os.path.join(savedir_path,basename.replace('.pgn',f'_{i}_analyzed.jsonl'))
    with open(jsonl_path,'wt') as f:
        for info in game_info:
            f.write(json.dumps(info)+'\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, required=True, help='Target directory to analyze')
    args = parser.parse_args()

    pgn_files = args.target

    b_list = []
    for weight_file in weight_files:
        w = Weights(os.path.join(maia_setting_path,weight_file))
        b = Backend(weights=w, backend="cuda-fp16")
        b_list.append(b)

    game_n = 0
    position_n = 0

    start = time.time()

    with open(pgn_files) as pgn:
        while True:
            move_list = []
            input_list = []
            for i in range(len(weight_files)):
                input_list.append([])
            g_list = []

            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            Black_name = game.headers["Black"]
            White_name = game.headers["White"]
            BlackELO = game.headers['BlackElo']
            WhiteELO = game.headers['WhiteElo']
            Result = game.headers["Result"]
            Site = game.headers['Site']
            Event = game.headers['Event']

            board = game.board()
            
            results = []
            for move_i, played_move in enumerate(game.mainline_moves()):
                position_n += 1

                g = GameState(moves=move_list)
                g_list.append(g)
                legal_moves = g.moves()

                rootInfo = {'rootInfo':{}}
                models_results = []
                for b_idx,b in enumerate(b_list):
                    input = g.as_input(b)
                    output = b.evaluate(input)
                    value = output[0].q()
                    
                    policy = output[-1].p_softmax(*g.policy_indices())
                    max_policy = max(policy)
                    
                    rootInfo['rootInfo'][f'{model_labels[b_idx]}_value'] = value
                    rootInfo['rootInfo'][f'{model_labels[b_idx]}_policy'] = policy
                    rootInfo['rootInfo'][f'{model_labels[b_idx]}_max_prior'] = max_policy
                    rootInfo['rootInfo']['turnNumber'] = move_i
                    
                    legal_move_list_batch = [move_list+[move] for move in legal_moves]
                    legal_game_states = [GameState(moves=lml) for lml in legal_move_list_batch]
                    legal_inputs = [gs.as_input(b) for gs in legal_game_states]
                    outputs = b.evaluate(*legal_inputs)
                    legal_moves_value_list = [o.q() for o in outputs]
                    legal_moves_prior_list = [policy[idx] for idx in range(len(legal_moves))]

                    models_results.append([legal_moves,legal_moves_value_list,legal_moves_prior_list])
                
                moveInfos = {'moveInfos':[]}
                for idx,move in enumerate(models_results[0][0]):
                    moveInfo = {'move':move}
                    for model_idx,model_result in enumerate(models_results):
                        moveInfo[f'{model_labels[model_idx]}_value'] = model_result[1][idx]
                        moveInfo[f'{model_labels[model_idx]}_prior'] = model_result[2][idx]
                    moveInfos['moveInfos'].append(moveInfo)
                move_list.append(played_move.uci())
                board.push(played_move)
                results.append({**moveInfos,**rootInfo})

            baseInfo = {
                'Site':Site,
                'history_moves':move_list,
                'PB':Black_name,'PW':White_name,
                'BR':BlackELO,'WR':WhiteELO,
                'Event':Event,'Result':Result,
            }
            game_info = [baseInfo] + results

            save_result(pgn_files,game_info,game_n)
            
            game_n += 1
            end = time.time()
            if game_n % 1000 == 0:
                print(game_n,format(end-start,'.1f'),'seconds No.ilegal')

if __name__ == "__main__":
    main()
