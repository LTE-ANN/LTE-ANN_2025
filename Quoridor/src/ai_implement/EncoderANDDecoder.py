#x,y를 인공지능이 알아먹을 수 있도록 하는 Encoder
#인공지능이 출력한 숫자를 x,y로 바꿔주는 Decoder
import numpy as np

class NotAllowPosException(Exception):
    def __init__(self, message="position not allowed"):
        self.message = message
        super().__init__(self.message)

class NotAllowMoveException(Exception):
    def __init__(self, message="move not allowed"):
        self.message = message
        super().__init__(self.message)

#인공지능이 출력한 block에 관련된 수를 좌표로 바꿔주는 decoder.
#인공지능이 출력한 array는 row_block 자리 (N-1)*(N-1)개 먼저, 그 다음 col_block 자리 (N-1)*(N-1)개 순서로 나열된 자리 중 위치
#decoder는 (x, y, direction) 형태로 출력
def block_decoder(n, N):
    if 0 <= n < (N-1)*(N-1):
        return n//(N-1), n%(N-1), 'row'
    elif (N-1)*(N-1) <= n < 2*(N-1)*(N-1):
        return n//(N-1)-(N-1), n%(N-1), 'col'
    raise NotAllowPosException

def block_encoder(x, y, direction, N):
    if(type(direction) == int):
        if direction ==0:
            direction = 'row'
        elif direction ==1:
            direction = 'col'
        else:
            raise NotAllowPosException

    if direction =='row':
        return x*(N-1) + y
    elif direction =='col':
        return x*(N-1) + y + (N-1)*(N-1)
    raise NotAllowPosException

#인공지능이 출력한 움직임에 관련된 수를 좌표로 바꿔주는 decoder
#decoder는 (x,y) 형태로 출력
def player_decoder(n):
    move_dict = {0:'w', 1:'a', 2:'s', 3:'d', None:None}
    if n in move_dict.keys():
        return move_dict[n]
    raise NotAllowMoveException

def player_encoder(n):
    move_dict = {'w':0, 'a':1, 's':2, 'd':3, None:None}
    if n in move_dict.keys():
        return move_dict[n]
    raise NotAllowMoveException

def player_gamestate_decoder(n,N):
    if 0 <= n < N*N:
        return n//N, n%N
    raise NotAllowPosException

'''def preprogressor(mcts_step_record: MCTSStepRecord, N):
    player_board_1 = np.zeros(N * N, dtype=int)
    player_board_1[mcts_step_record['state']['player_state_1']] = 1
    player_board_2 = np.zeros(N * N, dtype=int)
    player_board_2[mcts_step_record['state']['player_state_2']] = 1
    input_1 = [torch.tensor(mcts_step_record['state']['block_state_1'] + mcts_step_record['state']['block_state_2'],
                            dtype=torch.float32), torch.tensor(player_board_1 + player_board_2).unsqueeze(0)]
    input_2 = [torch.tensor(mcts_step_record['state']['block_state_2'] + mcts_step_record['state']['block_state_1'],
                            dtype=torch.float32), torch.tensor(player_board_2 + player_board_1).unsqueeze(0)]

    np_edge = np.array(list(map(lambda d: list(d.values()), mcts_step_record['edges'])))

    move_1 = np.bincount(np_edge[:, 0], weights=np_edge[:, 4])
    place_1 = np.bincount(np_edge[:, 1], weights=np_edge[:, 4])
    move_2 = np.bincount(np_edge[:, 2], weights=np_edge[:, 4])
    place_2 = np.bincount(np_edge[:, 3], weights=np_edge[:, 4])

    move_1, place_1, move_2, place_2 = move_1/np.sum(move_1), place_1/(np.sum(place_1)), move_2/np.sum(move_2), place_2/np.sum(place_2)

    return input_1, input_2, move_1, place_1, move_2, place_2, mcts_step_record['reward_1'], mcts_step_record['reward_2']'''