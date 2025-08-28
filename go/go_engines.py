import subprocess
import os
import sys
import json
import math
import platform

import pysgf
from gtp import parse_vertex

operating_system = platform.system()
katago_exe_path = 'The path to your KataGo.exe'
katago_setting_dir = 'The path to your KataGo directory'
katago_net_file = 'kata1-b18c384nbt-s9732312320-d4245566942.bin.gz'
katago_human_net_file = 'b18c384nbt-humanv0.bin.gz'


coord_alphabet = 'ABCDEFGHJKLMNOPQRST'
proper_first_moves = {}
def get_proper_first_moves(board_size: int):
    moves = []
    for i in range(int(board_size / 2), board_size):
        for j in range(board_size - i - 1, int(board_size / 2) + 1):
            proper_first_move = coord_alphabet[i] + str(board_size - j)
            moves.append(proper_first_move)
    return moves
proper_first_moves[9] = get_proper_first_moves(9)
proper_first_moves[13] = get_proper_first_moves(13)
proper_first_moves[19] = get_proper_first_moves(19)

class KataGoAnalysisEngine:
    def __init__(self, katago_analysis_cmd: str):
        self.analysis_process = subprocess.Popen(
            katago_analysis_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.katago_analysis_cmd = katago_analysis_cmd

        while True:
            # Analysis Engine starting...
            # ...
            # Started, ready to begin handling requests
            message = self.analysis_process.stderr.readline()
            if b'' == message:
                continue
            # print(message)
            if b'Started, ready to begin handling requests' in message:
                break

    def __del__(self):
        self.analysis_process.terminate()

    def send_query(self, query: dict):
        query_str = json.dumps(query) + '\n'
        self.analysis_process.stdin.write(query_str.encode())
        self.analysis_process.stdin.flush()

    def recv_one_response(self):
        response_line\
            = self.analysis_process.stdout.readline().decode().strip()
        return json.loads(response_line.replace('\r', ''))

class BaseGtpEngine:
    def __init__(self, exec_args: str, stderr_to = None):
        self.gtp_process = subprocess.Popen(
            exec_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=(open(os.devnull, 'w') if stderr_to is None else stderr_to)
        )

    def __del__(self):
        self.gtp_process.terminate()

    def send_and_recv(self, gtp_cmd):
        #import sys
        gtp_cmd += '\n'
        self.gtp_process.stdin.write(gtp_cmd.encode())
        self.gtp_process.stdin.flush()
        result = ''
        while True:
            data = self.gtp_process.stdout.readline().decode()
            if not data.strip():
                break
            result += data.replace('\r', '')
        return result

    def quit(self):
        self.gtp_process.communicate("quit\n")

    def name(self):
        name_str = self.send_and_recv('name')
        # eliminate '= ' at the front and '\n' at the end
        return name_str[2:-1]

    def boardsize(self, boardsize):
        self.send_and_recv('boardsize ' + str(boardsize))

    def komi(self, komi):
        self.send_and_recv('komi ' + str(komi))

    def clear_board(self):
        self.send_and_recv('clear_board')

    def play(self, gtp_color, gtp_move):
        self.send_and_recv('play ' + gtp_color + ' ' + gtp_move)

    def genmove(self, gtp_color):
        move_str = self.send_and_recv('genmove ' + gtp_color)
        # eliminate '= ' at the front and '\n' at the end
        return move_str[2:-1]

    def printsgf(self, sgf_path):
        self.send_and_recv('printsgf ' + sgf_path)

    def loadsgf(self, sgf_path, move_id: int=None):
        load_cmd = 'loadsgf ' + sgf_path
        if move_id is not None:
            load_cmd += ' ' + str(move_id)
        gtp_color_str = self.send_and_recv(load_cmd)
        return gtp_color_str[2:-1]

    def final_score(self):
        final_score = self.send_and_recv('final_score')
        return final_score[2:-1]

    # this command is specific for gtp_player.py
    def set_final_score(self, final_score):
        self.send_and_recv('set_final_score ' + final_score)

class LeelazGtpEngine(BaseGtpEngine):
    def undo(self):
        self.send_and_recv('undo')

    def clear_cache(self):
        self.send_and_recv('clear_cache')

    def get_policy_value(self):
        policy_value_str = self.send_and_recv('get_policy_value')
        # eliminate '= ' at the front and '\n' at the end
        policy_value_str = policy_value_str[2:-1]
        policy_value_list = []
        for s in policy_value_str.split(' '):
            policy_value_list.append(float(s))
        return policy_value_list[:-1], policy_value_list[-1]

    def show1dboard(self):
        board_1dstr = self.send_and_recv('show1dboard')
        # eliminate '= ' at the front and '\n' at the end
        return board_1dstr[2:-1]

    def get_prisoners(self, gtp_color):
        prisoner_num_str = self.send_and_recv('get_prisoners ' + gtp_color)
        # eliminate '= ' at the front and '\n' at the end
        return int(prisoner_num_str[2:-1])

class KataGoGtpEngine(BaseGtpEngine):
    def search_debug(self, gtp_color):
        move_str = self.send_and_recv('search_debug ' + gtp_color)
        # eliminate '= ' at the front and '\n' at the end
        return move_str[2:-1]

    def clear_cache(self):
        self.send_and_recv('clear_cache')

def append_humansl_results(board_size_tuple, katago_analysis, humansl_analysis, humansl_profile):
    katago_analysis[humansl_profile + '_policy'] = humansl_analysis['policy']
    katago_analysis['rootInfo'][humansl_profile + '_winrate'] = humansl_analysis['rootInfo']['winrate']
    katago_analysis['rootInfo'][humansl_profile + '_scoreLead'] = humansl_analysis['rootInfo']['scoreLead']
    for move_info in katago_analysis['moveInfos']:
        move_info[humansl_profile] = retrieve_data(
            humansl_analysis['policy'], board_size_tuple, move_info['move']
        )

def retrieve_data(data: list, board_size_tuple: tuple, gtp_vertex: str):
    if gtp_vertex.lower() == 'pass':
        return data[-1]
    # e.g., transform A1 to (1, 1)
    x, y = parse_vertex(gtp_vertex)
    # calculate coord_id so that the top-left is (0, 0) -> 0
    coord_id = (board_size_tuple[1] - y) * board_size_tuple[0] + (x - 1)
    return data[coord_id]

def calculate_distance(gtp_move_1: str, gtp_move_2: str, type: str = None):
    x1, y1 = parse_vertex(gtp_move_1)
    x2, y2 = parse_vertex(gtp_move_2)
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    if type is None:
        # Euclidean distance
        return math.sqrt(dx ** 2 + dy ** 2)
    if type == 'remi':
        return dx + dy + max(dx, dy)
    return 0

def get_matrix_output(data: list, board_size: tuple, fmt: str):
    output = ''
    for i, d in enumerate(data):
        output += (format(d, fmt) + ' ')
        if (i + 1) % board_size[0] == 0:
            output += '\n'
    return output

def get_comment(analysis: dict, board_size: tuple,
        interested_fields_and_display_format: dict, is_simplified: bool = False):
    comment = ''
    move_infos = analysis['moveInfos']
    interested_fields = interested_fields_and_display_format.keys()
    comment += (' | '.join(interested_fields) + ' | \n')
    for move_info in move_infos:
        for interested_field in interested_fields:
            if interested_field in move_info:
                fmt = interested_fields_and_display_format[interested_field]
                if fmt is None:
                    value_str = str(move_info[interested_field])
                else:
                    value_str = format(move_info[interested_field], fmt)
                if interested_field in ['scoreLead', 'utility']:
                    if move_info[interested_field] >= 0:
                        value_str = '+' + value_str
            else:
                value_str = ''
            comment += (value_str.rjust(len(interested_field), '.') + ' | ')
        comment += '\n'

    if not is_simplified:
        if 'ownership' in analysis:
            comment += '\nKataGo ownership\n'
            comment += get_matrix_output(analysis['ownership'], board_size, fmt='+.02f')
        
        if 'policy' in analysis:
            comment += '\nKataGo policy\n'
            comment += get_matrix_output(analysis['policy'], board_size, fmt='.03f')
            comment += '\n' # this newline is for PASS

        for key in analysis:
            if key.endswith('_policy'):
                comment += '\n' + key + '\n'
                comment += get_matrix_output(analysis[key], board_size, fmt='.03f')
                comment += '\n' # this newline is for PASS
        
    return comment

def get_sgf_labels(analysis: dict, board_size: tuple):
    move_infos = analysis['moveInfos']
    labels = []
    for move_info in move_infos:
        order = move_info['order']
        if 'prior' in move_info:
            p_str = str(int(move_info['prior'] * 1000))
        else:
            p_str = 'NA'
        label = pysgf.Move.from_gtp(move_info['move']).sgf(board_size)\
            + ':' + str(order) + '\n' + p_str
        labels.append(label)
    return labels

def calculate_score_loss(analysis: dict):
    move_infos = analysis['moveInfos']
    max_lead = -sys.maxsize
    for move_info in move_infos:
        # print(move_info['visits'])
        if move_info['visits'] > 10:
            max_lead = max(max_lead, move_info['scoreLead'])
    # print(max_lead)
    for move_info in move_infos:
        move_info['scoreLoss'] = max_lead - move_info['scoreLead']  

def calculate_score_gain(analysis: dict):
    move_infos = analysis['moveInfos']
    pass_lead = None
    for move_info in move_infos:
        if move_info['move'] == 'pass':
            pass_lead = move_info['scoreLead']
    if pass_lead is None:
        return
    for move_info in move_infos:
        move_info['scoreGain'] = move_info['scoreLead'] - pass_lead

def calculate_orderV_orderLCB(analysis: dict):
    move_infos = analysis['moveInfos']
    move_infos.sort(key=lambda x:x['visits'], reverse=True)
    for i, move_info in enumerate(move_infos):
        move_info['orderV'] = i
    move_infos.sort(key = lambda x:x['lcb'], reverse=True)
    for i, move_info in enumerate(move_infos):
        move_info['orderL'] = i
    move_infos.sort(key = lambda x:x['order'])

def get_game_length(sgf: pysgf.SGFNode):
    if len(sgf.children) == 0:
        return 0
    node = sgf.children[0]
    length = 0
    while True:
        length += 1
        if len(node.children) == 0:
            break
        node = node.children[0]
    return length

def get_initial_stones(sgf_root: pysgf.SGFNode):
    initial_stones = []
    for initial_stone in sgf_root.placements:
        initial_stones.append([initial_stone.player, initial_stone.gtp()])
    return initial_stones

def get_history_moves(sgf_root: pysgf.SGFNode):
    moves = []
    if len(sgf_root.children) > 0:
        node = sgf_root.children[0]
        while True:
            # ToDo: need to consider branches in sgf?
            moves.append([node.move.player, node.move.gtp()])
            if len(node.children) == 0:
                break
            node = node.children[0]
    return moves

def parse_exec_args(file_name: str):
    with open(file_name, 'r') as f:
        args_line = f.readline().strip()
        args = args_line.split(' ')
        return args

