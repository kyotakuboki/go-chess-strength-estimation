"""
./sgf_analyzer.py --help
This shows the explanations of program arguments.

a typical example of execution:
./sgf_analyzer.py -t sgf/ -n 1000 -e 200
This analyzes all game records in sgf/ (under the same directory)
 with 1000 simulations for each position
 and 200 additional simulations for the played moves.

search each candidate move with some simulations
candidate move set: priorLZ top 15 + priorKT top 3 + PASS + played + o/x (if any)
number of simulations per move: -n
./sgf_analyzer.py -t sgf/ -n 100 -pr

options:
-p for doing additional search for PASS when -e is used
-last for only searching for the last move (mainly for analyzing problems from books with reconstructed move order)
--no-move-owner for not including each move's ownership
-g [int] for assigning the maximum number of games to analyze
--override for ignoring previously analyzed results
"""

import pysgf
import json
import time
import gzip
import os
import argparse
import copy
from gtp import gtp_vertex

from go_engines import katago_exe_path, katago_setting_dir, katago_net_file, katago_human_net_file
from go_engines import KataGoAnalysisEngine, LeelazGtpEngine
from go_engines import append_humansl_results
from go_engines import get_comment, get_sgf_labels
from go_engines import calculate_score_gain
from go_engines import get_initial_stones, get_history_moves
from go_engines import retrieve_data

class OneGameAnalysis:
    def __init__(self, sgf_path: str = '', idx: int = 0):
        self.sgf_path = sgf_path
        self.idx = idx
        self.sgf_content = None
        self.move_num = 0

        self.move_space = []

        self.initial_stones = None
        self.history_moves = None

        # a single query for analyzing all move (analyzeTurns)
        # the results of each move are in different dict
        self.analysis_query = {}
        self.analysis_results = []

        # the board string of each move (before move)
        self.board_strings = []

        # analyze the played moves or PASS at each turn by extra queries
        # the temporary results are merged into 'self.analysis_results'
        self.allow_moves_search_failed = ''
        self.played_move_queries = []
        self.played_move_results = []
        self.pass_move_queries = []
        self.pass_move_results = []

        # for each move, those o or x is included in sgf
        # [[], []]
        self.additional_move_candidates = []

    def merge_extra_search_results(self, target: str):
        # target: currently either 'played' or 'pass'
        if target == 'played':
            extra_results = self.played_move_results
        elif target == 'pass':
            extra_results = self.pass_move_results
        else:
            return

        for i in range(self.move_num):
            # only 'moveInfos': [] and 'turnNumber' added manually when only analyzing the last move
            if len(extra_results[i]) == 2:
                continue
            if len(extra_results[i]['moveInfos']) == 0:
                self.allow_moves_search_failed += str(i + 1) + ' '
                continue
            played_move_extra_info = extra_results[i]['moveInfos'][0]
            played_move_info = None
            for move_info in self.analysis_results[i]['moveInfos']:
                if move_info['move'] == played_move_extra_info['move']:
                    played_move_info = move_info
                    break
            if played_move_info is None:
                played_move_extra_info['order'] = 'n/a'
                self.analysis_results[i]['moveInfos'].append(played_move_extra_info)
            else:
                total_visits = played_move_info['visits'] + played_move_extra_info['visits']
                merge_fileds = ['scoreLead', 'scoreStdev', 'winrate']
                for f in merge_fileds:
                    played_move_info[f] = (played_move_info[f] * played_move_info['visits']\
                        + played_move_extra_info[f] * played_move_extra_info['visits']) / total_visits
                played_move_info['visits'] = total_visits

class SgfAnalyzer:
    def __init__(self,katago_analysis_engine: KataGoAnalysisEngine,
                      kata_human_analysis_engine: KataGoAnalysisEngine):

        self.katago_analysis_engine = katago_analysis_engine
        self.kata_human_analysis_engine = kata_human_analysis_engine
        
        # self.analysis_target_paths = []
        self.analysis_target_path = None
        self.analysis_target_list = []
        self.save_dir = ''

        self.comment_interested_fields_and_display_format = {
            'order': None, 'move': None,
            'prior': '.03f',
            'visits': None, 'winrate': '.03f', 
            'scoreLead': '.02f', 
            'scoreStdev': '.02f', 
            'scoreGain': '.02f'
        }

        self.current_game_and_analysis = None # OneGameAnalysis object
        self.no_move_owner = False
        self.game_num_max = None
        self.is_overriding = False
        self.skip_jsonl = False
        self.skip_analyzed_sgf = False

    def has_been_analyzed(self, sgf_path: str):
        base_name = os.path.basename(sgf_path)
        analyzed_sgf_path = os.path.join(self.save_dir, base_name.replace('.sgf', '_analyzed.sgf'))
        analyzed_jsonl_path = os.path.join(self.save_dir, base_name.replace('.sgf', '_analyzed.jsonl.gz'))
        return os.path.exists(analyzed_sgf_path) or os.path.exists(analyzed_jsonl_path)

    def set_analysis_target(self, target_path: str):
        if os.path.exists(target_path) == False:
            print(target_path, 'does not exist.')
            return
        with open(target_path, 'r', encoding='UTF-8') as f:
            content = f.readlines()
        self.analysis_target_list = content
        self.analysis_target_path = target_path
        base_dir = os.path.dirname(os.path.abspath(target_path))
        self.save_dir = os.path.join(
            base_dir, 
            f'{os.path.basename(self.analysis_target_path)[:-len(".sgf")]}_analyzed')
        if os.path.exists(self.save_dir) == False:
            os.mkdir(self.save_dir)

    def load_sgf(self, id: int):
        content = self.analysis_target_list[id]
        basename = os.path.basename(self.analysis_target_path)
        basename = basename.replace('.sgf', f'_{id}_analyzed.sgf')
        jsonl_path = os.path.join(self.save_dir, basename.replace('.sgf', f'_{id}.jsonl.gz'))
        
        if os.path.exists(jsonl_path) and not self.is_overriding:
            print(id,'has been analyzed')
            return 'True'
        
        game_analysis = OneGameAnalysis(
            self.analysis_target_path,id)
        # for the pysgf library, tt can only be used as PASS for 19x19
        # because it judges by alphabet[board size]
        # replacing with empty works for board sizes <= 19
        content = content.replace('B[tt]', 'B[]')
        content = content.replace('W[tt]', 'W[]')
        sgf = pysgf.SGF.parse(content)
        # remove existing comment and LB
        # except for Turn number 0, which may contain some basic info
        node = sgf
        is_move0 = True
        while len(node.children) > 0:
            if 'AB' in node.properties.keys():
                return None
            if 'C' in node.properties:
                if is_move0 == True:
                    is_move0 == False
                else:
                    node.set_property('C', '')
            if 'LB' in node.properties:
                node.set_property('LB', '')
            additional_candidates = []
            for mark in ['CR', 'MA']:
                if mark not in node.properties:
                    continue
                for cand in node.properties[mark]:
                    additional_candidates.append(
                        pysgf.Move.from_sgf(cand, player='B', board_size=sgf.board_size).gtp()
                    )
            game_analysis.additional_move_candidates.append(additional_candidates)
            node = node.children[0]

        game_analysis.sgf_content = sgf
        game_analysis.initial_stones = get_initial_stones(sgf)
        game_analysis.history_moves = get_history_moves(sgf)
        game_analysis.move_space = []
        for x in range(sgf.board_size[0]):
            for y in range(sgf.board_size[1]):
                game_analysis.move_space.append(gtp_vertex((x + 1, y + 1)))
        game_analysis.move_space.append('pass')
        self.current_game_and_analysis = game_analysis

        return 'Done'

    def get_katago_analysis_query(self, id: int, max_visits: int = 500):
        sgf = self.current_game_and_analysis.sgf_content

        initial_stones = self.current_game_and_analysis.initial_stones
        history_moves = self.current_game_and_analysis.history_moves

        katago_analysis = {}
        katago_analysis['id'] = str(id)
        
        katago_analysis['moves'] = history_moves
        if len(initial_stones) > 0:
            katago_analysis['initialStones'] = initial_stones

        katago_analysis['rules'] = 'japanese'
        if 'RU' in sgf.properties:
            sgf_rule_str = sgf.ruleset.lower()
            katago_analysis['rules'] = sgf_rule_str
            if sgf_rule_str == 'jp' or sgf_rule_str == 'japanese':
                katago_analysis['rules'] = 'japanese'
            elif sgf_rule_str == 'cn' or sgf_rule_str == 'chinese':
                katago_analysis['rules'] = 'chinese'
            elif sgf_rule_str == 'kr' or sgf_rule_str == 'korean':
                katago_analysis['rules'] = 'korean'
            elif sgf_rule_str == 'nz' or sgf_rule_str == 'new-zealand'\
                    or sgf_rule_str == 'new zealand':
                katago_analysis['rules'] = 'new-zealand'
            elif sgf_rule_str == 'aga':
                katago_analysis['rules'] = 'aga'
            else:
                print('Unknown rule', sgf.ruleset, 'Use Japanese rules as default.')

        katago_analysis['komi'] = 6.5
        
        katago_analysis['boardXSize'] = sgf.board_size[0]
        katago_analysis['boardYSize'] = sgf.board_size[1]

        if len(history_moves) > 0:
            if history_moves[0][0].lower() == 'w':
                katago_analysis['initialPlayer'] = 'W'

        katago_analysis['maxVisits'] = max_visits
        katago_analysis['includeOwnership'] = True
        if self.no_move_owner == False:
            katago_analysis['includeMovesOwnership'] = True
        katago_analysis['includePolicy'] = True
        katago_analysis['overrideSettings'] = {
            'reportAnalysisWinratesAs':'SIDETOMOVE'
        }
        
        return katago_analysis

    def send_normal_analysis_queries_to_katago(self, id: int, max_visits: int, extra_visits: int, also_pass: bool, only_last_move: bool):
        game_analysis = self.current_game_and_analysis
        base_query = self.get_katago_analysis_query(id, max_visits)
        full_game_query = copy.deepcopy(base_query)
        if len(game_analysis.history_moves) > 0 and not only_last_move:
            full_game_query['analyzeTurns'] = [i for i in range(len(game_analysis.history_moves))]
        if only_last_move:
            full_game_query['analyzeTurns'] = [len(game_analysis.history_moves) - 1]
        game_analysis.analysis_query = full_game_query
        game_analysis.move_num = max(len(game_analysis.history_moves), 1)
        self.katago_analysis_engine.send_query(full_game_query)

        game_analysis.played_move_queries = []
        if extra_visits is not None:
            for j, move in enumerate(game_analysis.history_moves):
                if only_last_move and j != (len(game_analysis.history_moves) - 1):
                    game_analysis.played_move_queries.append({})
                    continue
                played_move_query = copy.deepcopy(base_query)
                played_move_query['id'] += '_' + str(j)
                played_move_query['moves'] = played_move_query['moves'][0:j]
                played_move_query['maxVisits'] = extra_visits
                played_move_query['allowMoves'] = [{'player': move[0], 'moves': [move[1]], 'untilDepth': 1}]
                game_analysis.played_move_queries.append(copy.deepcopy(played_move_query))
                self.katago_analysis_engine.send_query(played_move_query)
        if extra_visits is not None and also_pass:
            for j, move in enumerate(game_analysis.history_moves):
                if only_last_move and j != (len(game_analysis.history_moves) - 1):
                    game_analysis.pass_move_queries.append({})
                    continue
                pass_move_query = copy.deepcopy(base_query)
                pass_move_query['id'] += '_' + str(j) + 'p'
                pass_move_query['moves'] = pass_move_query['moves'][0:j]
                pass_move_query['maxVisits'] = extra_visits
                pass_move_query['allowMoves'] = [{'player': move[0], 'moves': ['pass'], 'untilDepth': 1}]
                game_analysis.pass_move_queries.append(copy.deepcopy(pass_move_query))
                self.katago_analysis_engine.send_query(pass_move_query)
        
    def recv_normal_analysis_responses_from_katago(self, has_played_move_analysis: bool, also_pass: bool, only_last_move: bool):
        game_analysis = self.current_game_and_analysis
        has_finished = False
        recv_num = 0
        while has_finished == False:
            result = self.katago_analysis_engine.recv_one_response()
            recv_num += 1
            id_str = result['id']
            if 'p' in id_str:
                # pass
                target_results = game_analysis.pass_move_results
            elif '_' in id_str:
                # played move
                target_results = game_analysis.played_move_results
            else:
                target_results = game_analysis.analysis_results
            target_results.append(result)

            move_num = game_analysis.move_num
            if 'error' in result or 'warning' in result:
                if len(game_analysis.analysis_results) == 0:
                    continue
                if has_played_move_analysis and len(game_analysis.played_move_results) != move_num:
                    continue
                if has_played_move_analysis and also_pass and len(game_analysis.pass_move_results) != move_num:
                    continue
                return
            if len(game_analysis.analysis_results) == move_num:
                if has_played_move_analysis and\
                        len(game_analysis.played_move_results) != move_num:
                    continue
                if has_played_move_analysis and also_pass and\
                        len(game_analysis.pass_move_results) != move_num:
                    continue
                has_finished = True
            elif only_last_move:
                if recv_num == 1 + int(has_played_move_analysis) + int(also_pass):
                    for i in range(len(game_analysis.history_moves) - 1):
                        game_analysis.analysis_results.append({'moveInfos': [], 'turnNumber': i})
                        if has_played_move_analysis:
                            game_analysis.played_move_results.append({'moveInfos': [], 'turnNumber': i})
                        if has_played_move_analysis and also_pass:
                            game_analysis.pass_move_results.append({'moveInfos': [], 'turnNumber': i})
                    has_finished = True

        game_analysis.analysis_results.sort(key=lambda x:x['turnNumber'])
        if has_played_move_analysis:
            game_analysis.played_move_results.sort(key=lambda x:x['turnNumber'])
            game_analysis.merge_extra_search_results(target='played')
        if has_played_move_analysis and also_pass:
            game_analysis.pass_move_results.sort(key=lambda x:x['turnNumber'])
            game_analysis.merge_extra_search_results(target='pass')

    def get_katago_humansl_policy_and_value(self, id: int, only_last_move: bool, humansl_profile: str):
        game_analysis = self.current_game_and_analysis
        if 'error' in game_analysis.analysis_results[0] or 'warning' in game_analysis.analysis_results[0]:
            return

        base_query = self.get_katago_analysis_query(id, 1)
        full_game_query = copy.deepcopy(base_query)
        if len(game_analysis.history_moves) > 0 and not only_last_move:
            full_game_query['analyzeTurns'] = [i for i in range(len(game_analysis.history_moves))]
        if only_last_move:
            full_game_query['analyzeTurns'] = [len(game_analysis.history_moves) - 1]
        game_analysis.analysis_query = full_game_query
        game_analysis.move_num = max(len(game_analysis.history_moves), 1)
        full_game_query['overrideSettings']['humanSLProfile'] = humansl_profile
        
        self.kata_human_analysis_engine.send_query(full_game_query)

        humansl_results = []
        has_finished = False
        recv_num = 0
        while has_finished == False:
            result = kata_human_analysis_engine.recv_one_response()
            recv_num += 1
            humansl_results.append(result)

            if 'error' in result or 'warning' in result:
                if len(humansl_results) == 0:
                    continue
                return
            if len(humansl_results) == len(game_analysis.history_moves):
                has_finished = True
        humansl_results.sort(key=lambda x:x['turnNumber'])

        board_size_tuple = self.current_game_and_analysis.sgf_content.board_size
        for i in range(len(humansl_results)):
            append_humansl_results(board_size_tuple, game_analysis.analysis_results[i], humansl_results[i], humansl_profile)

    def get_promissing_to_analyze_candidates(self, katago_p: list, humansl_p_list: list, played_move: str, additional_candidates: list):
        board_size_tuple = self.current_game_and_analysis.sgf_content.board_size
        candidates = []

        move_and_priorKT_list = []
        move_and_humansl_lists = []
        for _ in humansl_p_list:
            move_and_humansl_lists.append([])
        for move in self.current_game_and_analysis.move_space:
            p = retrieve_data(katago_p, board_size_tuple, move)
            if p > 0:
                move_and_priorKT_list.append({'move':move,'priorKT':p})
            for idx,humansl in enumerate(humansl_p_list):
                p = retrieve_data(humansl, board_size_tuple, move)
                if p > 0:
                    move_and_humansl_lists[idx].append({'move':move,'prior':p})
        move_and_priorKT_list.sort(key=lambda x:x['priorKT'], reverse=True)
        
        # first, get top-20 from priorKT
        for i in range(min(len(move_and_priorKT_list),20)):
            move = move_and_priorKT_list[i]['move']
            if move not in candidates:
                candidates.append(move)
        
        # second, get top-20 from human prior 
        for move_and_humansl_list in move_and_humansl_lists:
            move_and_humansl_list.sort(key=lambda x:x['prior'],reverse=True)
            for i in range(min(len(move_and_humansl_list),20)):
                move = move_and_humansl_list[i]['move']
                if move not in candidates:
                    candidates.append(move)
        
        # always include pass in candiate
        if 'pass' not in candidates:
            candidates.append('pass')

        # include the played move
        if played_move not in candidates:
            candidates.append(played_move)

        # include additional candidates (o and x)
        for cand in additional_candidates:
            if cand not in candidates:
                candidates.append(cand)

        return candidates

    def send_and_recv_promissing_move_quneries_to_katago(self, id: int, visits_per_candidate: int, only_last_move: bool):
        game_analysis = self.current_game_and_analysis

        self.send_normal_analysis_queries_to_katago(id, 1, None, False, only_last_move)
        self.recv_normal_analysis_responses_from_katago(False, False, only_last_move)
        if self.katago_humansl_settings is not None:
            for human_profile in self.katago_humansl_settings:
                self.get_katago_humansl_policy_and_value(id, only_last_move, human_profile)
            self.kata_human_analysis_engine.send_query({'id':'c','action':'clear_cache'})
            self.kata_human_analysis_engine.recv_one_response()
        if 'error' in game_analysis.analysis_results[0] or 'warning' in game_analysis.analysis_results[0]:
            return
        base_query = self.get_katago_analysis_query(id, visits_per_candidate)
        for i, move in enumerate(game_analysis.history_moves):
            # get a list of candidates: leela, kt, played, pass
            katago_p = game_analysis.analysis_results[i]['policy']
            humansl_p_list = []
            for humansl_p in self.katago_humansl_settings:
                humansl_p_list.append(game_analysis.analysis_results[i][f'{humansl_p}_policy'])
            
            candidates =\
                self.get_promissing_to_analyze_candidates(
                    katago_p,humansl_p_list,
                    move[1], game_analysis.additional_move_candidates[i])
            game_analysis.analysis_results[i]['moveInfos'].clear()
            for candidate in candidates:
                candidate_move_query = copy.deepcopy(base_query)
                candidate_move_query['id'] += '_' + candidate
                candidate_move_query['moves'] = candidate_move_query['moves'][0:i]
                candidate_move_query['allowMoves'] = [{'player': move[0], 'moves': [candidate], 'untilDepth': 1}]
                self.katago_analysis_engine.send_query(candidate_move_query)
                result = self.katago_analysis_engine.recv_one_response()
                if 'error' in result or 'warning' in result:
                    continue
                if len(result['moveInfos']) == 0:
                    continue
                game_analysis.analysis_results[i]['moveInfos'].append(result['moveInfos'][0])
            game_analysis.analysis_results[i]['moveInfos'].sort(key=lambda x:x['scoreLead'], reverse=True)
            for j in range(len(game_analysis.analysis_results[i]['moveInfos'])):
                game_analysis.analysis_results[i]['moveInfos'][j]['order'] = j
            self.katago_analysis_engine.send_query({'id':'c','action':'clear_cache'});
            self.katago_analysis_engine.recv_one_response()

    def save_results(self, id: int,
                     only_last_move: bool, skip_jsonl:bool,skip_analyzed_sgf:bool, 
                     leela_zero_network_variant: str=None,
                     ):
        basename = os.path.basename(self.current_game_and_analysis.sgf_path)
        basename = basename.replace('.sgf', f'_{id}_analyzed.sgf')
        jsonl_path = os.path.join(self.save_dir, basename.replace('.sgf', f'_{id}.jsonl.gz'))
        if not skip_jsonl:
            with gzip.open(jsonl_path, 'wt') as f:
                basic_info = {
                    'sgf_path': os.path.abspath(self.current_game_and_analysis.sgf_path),
                    'board_size': self.current_game_and_analysis.sgf_content.board_size,
                    'initial_stones': self.current_game_and_analysis.initial_stones,
                    'history_moves': self.current_game_and_analysis.history_moves
                }
                f.write(json.dumps(basic_info) + '\n')
                for analysis_result in self.current_game_and_analysis.analysis_results:
                    f.write(json.dumps(analysis_result) + '\n')
                    
        if 'error' in self.current_game_and_analysis.analysis_results[0]\
                or 'warning' in self.current_game_and_analysis.analysis_results[0]:
            print('Error/Warning happened when analyzing', self.current_game_and_analysis.sgf_path)
            if not skip_jsonl:
                print(json.dumps(self.current_game_and_analysis.analysis_results[0]))
            return jsonl_path
        if self.current_game_and_analysis.allow_moves_search_failed != '':
            print(self.current_game_and_analysis.sgf_path + ': The search of allowMoves failed at ' +\
                self.current_game_and_analysis.allow_moves_search_failed + 'move(s)')
            if not skip_jsonl:
                with open(os.path.join(self.save_dir, 'error.txt'), 'a') as f:
                    f.write(self.current_game_and_analysis.sgf_path + ' '
                        + self.current_game_and_analysis.allow_moves_search_failed + '\n')
            return 
        if not skip_analyzed_sgf:
            sgf = self.current_game_and_analysis.sgf_content
            node = sgf
            for i in range(self.current_game_and_analysis.move_num + 1):
                if i == 0 and node.get_property('C') is not None:
                    comment = node.get_property('C')
                else:
                    comment = '\nTurn number: ' + str(i) + '\n'
                # using 'i' feels more like position analysis (which move to play)
                # using 'i - 1' feels more like move analysis (how good was the move)
                comment += '' if i == 0 or (only_last_move and i != self.current_game_and_analysis.move_num) else\
                    get_comment(self.current_game_and_analysis.analysis_results[i - 1], sgf.board_size, 
                        self.comment_interested_fields_and_display_format
                    )
                if leela_zero_network_variant is not None:
                    comment += 'leela human network: ' + leela_zero_network_variant + '\n'
                node.set_property('C', comment)
                labels = [] if i == 0 else\
                    get_sgf_labels(self.current_game_and_analysis.analysis_results[i - 1], sgf.board_size)
                node.set_property('LB', labels)
                if len(node.children) > 0:
                    node = node.children[0]
            
            sgf_path = os.path.join(self.save_dir, basename)
            with open(sgf_path, 'w') as f:
                f.write(sgf.sgf() + '\n')
        print('Finished (', id + 1, '/',
            len(self.analysis_target_list), '): ', self.current_game_and_analysis.sgf_path, sep='')

    def calculate_move2idx(self, move:str):
        coord_alphabet = 'ABCDEFGHJKLMNOPQRST'
        if move == 'pass':
            return -1
        else:
            x_axis = coord_alphabet.index(move[0])
            y_axis = self.current_game_and_analysis.sgf_content.board_size[1] - int(move[1:])
            return self.current_game_and_analysis.sgf_content.board_size[1]*y_axis + x_axis

    def analyze_records(self, args):
        self.no_move_owner = args.no_move_owner
        self.game_num_max = args.game_num_max
        self.is_overriding = args.override
        self.skip_jsonl = args.skip_jsonl
        self.skip_analyzed_sgf = args.skip_analyzed_sgf
        if args.also_pass == False and args.allow_promissing_move == False:
            if 'scoreGain' in self.comment_interested_fields_and_display_format:
                self.comment_interested_fields_and_display_format.pop('scoreGain')
        self.set_analysis_target(args.target)
        self.katago_humansl_settings = None
        if args.katago_humansl_settings is not None:
            self.katago_humansl_settings = args.katago_humansl_settings.split('+')
            if 'scoreGain' in self.comment_interested_fields_and_display_format:
                self.comment_interested_fields_and_display_format.pop('scoreGain')
            if 'scoreStdev' in self.comment_interested_fields_and_display_format:
                self.comment_interested_fields_and_display_format.pop('scoreStdev')
            for setting in self.katago_humansl_settings:
                self.comment_interested_fields_and_display_format[setting] = '.05f'
        for i in range(len(self.analysis_target_list)):
            a = self.load_sgf(i)
            if a == None:
                print('error : in AB')
                continue
            elif a == 'True':
                continue
            self.get_policy_value_boardstr_from_leelaz()

            if args.allow_promissing_move:
                self.send_and_recv_promissing_move_quneries_to_katago(i, args.max_visits, args.only_last_move)
            else:
                self.send_normal_analysis_queries_to_katago(i, args.max_visits, args.extra_visits, args.also_pass, args.only_last_move)
                self.recv_normal_analysis_responses_from_katago((args.extra_visits is not None), args.also_pass, args.only_last_move)
            if self.katago_humansl_settings is not None:
                for human_profile in self.katago_humansl_settings:
                    self.get_katago_humansl_policy_and_value(i, args.only_last_move, human_profile)
                self.kata_human_analysis_engine.send_query({'id':'c','action':'clear_cache'})
                self.kata_human_analysis_engine.recv_one_response()
            self.append_leela_results()
            self.do_extra_calculation()
            self.save_results(i, args.also_pass, args.only_last_move, 
                              args.skip_jsonl,args.skip_analyzed_sgf, 
                              args.leela_zero_network_variant)
            self.katago_analysis_engine.send_query({'id':'c','action':'clear_cache'})
            self.katago_analysis_engine.recv_one_response()

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target', type=str,
    help='the target (an sgf file or a directory) to analyze')
parser.add_argument('-n', '--max-visits', type=int,
    help='the number of simulations to analyze each position')
parser.add_argument('-e', '--extra-visits', type=int, default=None,
    help='the number of simulations to additionally analyze the played move at each position')
parser.add_argument('-p', '--also-pass', action='store_true',
    help='with this argument, PASS is analyzed for each turn with extra-visits simulations')
parser.add_argument('-pr', '--allow-promissing-move', action='store_true',
    help='with this argument, moves from priorLZ/priorKT/played/PASS are analyzed by allowMoves')
parser.add_argument('-last', '--only-last-move', action='store_true',
    help='with this argument, only the last move is analyzed')
parser.add_argument('--no-move-owner', action='store_true',
    help='with this argument, KataGo query will not contain includeMovesOwnership')
parser.add_argument('-g', '--game-num-max', type=int, default=None,
    help='if assigned, only newly analyze game_num_max games')
parser.add_argument('--override', action='store_true',
    help='with this argument, sgf file(s) is analyzed no matter it has been analyzed or not')
parser.add_argument('--skip-jsonl',action='store_true',
    help='with this argument, only save as "*_analyzed.sgf" file')
parser.add_argument('--skip-analyzed-sgf',action='store_true',
    help='with this argument, only save as "*_analyzed.jsonl.gz" file')

parser.add_argument('-kth', '--katago-humansl-settings', type=str, default=None,
    help='the desired setting(s) for the humanSL model separated by plus, e.g., rank_5k+rank_1d')

args = parser.parse_args()

katago_analysis_cmd = [
    katago_exe_path,
    'analysis',
    '-model',
    katago_setting_dir + katago_net_file,
    '-config',
    'analysis_deterministic_analyzer.cfg'
]

kata_analysis_engine = KataGoAnalysisEngine(
    katago_analysis_cmd=katago_analysis_cmd
)

katago_human_analysis_cmd = [
    katago_exe_path,
    'analysis',
    '-model',
    katago_setting_dir + katago_human_net_file,
    '-config',
    'analysis_deterministic_analyzer.cfg',
]

kata_human_analysis_engine = KataGoAnalysisEngine(
    katago_analysis_cmd=katago_human_analysis_cmd
)

start = time.time()
analyzer = SgfAnalyzer(kata_analysis_engine, kata_human_analysis_engine)
analyzer.analyze_records(args)
end = time.time()
print('Analyzing', args.target, 'cost', format(end - start, '.1f'), 'seconds')
