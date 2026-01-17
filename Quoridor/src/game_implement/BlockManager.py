from collections import deque
from src.ai_implement.EncoderANDDecoder import block_decoder, block_encoder
import numpy as np
import torch
import pygame
import time

from src.game_implement.Block import RowBlock, ColBlock, RowBlock_for_ai, ColBlock_for_ai
from src.game_implement.Timer import Timer
from src.configs import *


class BlockManager:
    def __init__(self, n, cool_time = 1):
        self.row_blocks = [[RowBlock(i,j,0) for j in range(n-1)] for i in range(n-1)]
        self.col_blocks = [[ColBlock(i,j,0) for j in range(n-1)] for i in range(n-1)]
        self.square_images = [[pygame.transform.scale(pygame.image.load(area_image_path), (HEIGHT, HEIGHT))
                               for i in range(n)] for j in range(n)]
        self.N = n
        self.last_time = time.time()-1
        self.cool = cool_time#쿨타임
        self.timer = Timer(block_timer_image_dir, 9, "block_timer", TIMER_SIZE, TIMER_SIZE, TIMER_SIZE*0.2, SURFACE_HEIGHT-TIMER_SIZE*1.2)

        self._V = n * n
        self._adj_bits = [0] * self._V  # 각 노드의 이웃을 비트셋(int)으로 저장
        self._graph_dirty = True

    def _idx(self, x, y):
        return x * self.N + y


    def _mark_dirty(self):
        self._graph_dirty = True

    def _rebuild_graph_bits(self):
        N = self.N
        V = self._V
        adj = [0] * V

        def add_edge(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # 수평 이동 엣지: (x,y) <-> (x+1,y)
        # move_allow_d와 동일한 차단 규칙을 "엣지 단위"로 적용
        for y in range(N):
            for x in range(N - 1):
                blocked = False
                # col_blocks[x][y-1] 또는 col_blocks[x][y]가 있으면 막힘
                if y > 0 and self.col_blocks[x][y - 1].occupied:
                    blocked = True
                if y < N - 1 and self.col_blocks[x][y].occupied:
                    blocked = True
                if not blocked:
                    u = self._idx(x, y)
                    v = self._idx(x + 1, y)
                    add_edge(u, v)

        # 수직 이동 엣지: (x,y) <-> (x,y+1)
        # move_allow_s와 동일한 차단 규칙을 "엣지 단위"로 적용
        for y in range(N - 1):
            for x in range(N):
                blocked = False
                # row_blocks[x-1][y] 또는 row_blocks[x][y]가 있으면 막힘
                if x > 0 and self.row_blocks[x - 1][y].occupied:
                    blocked = True
                if x < N - 1 and self.row_blocks[x][y].occupied:
                    blocked = True
                if not blocked:
                    u = self._idx(x, y)
                    v = self._idx(x, y + 1)
                    add_edge(u, v)

        self._adj_bits = adj
        self._graph_dirty = False

    #후보 벽이 제거하는 "딱 2개 엣지"의 endpoint 쌍 계산 ---------
    # 네 코드 기준 direction:
    #  - 'row' (가로 벽): 세로 이동(위/아래)을 2칸(가로로 길이 2) 막는다
    #  - 'col' (세로 벽): 가로 이동(좌/우)을 2칸(세로로 길이 2) 막는다
    def _edges_for_wall(self, x, y, direction):
        N = self.N
        if not (0 <= x < N - 1 and 0 <= y < N - 1):
            return None

        if direction == 'row':
            # (x,y)-(x,y+1), (x+1,y)-(x+1,y+1)
            return [
                (self._idx(x, y),     self._idx(x, y + 1)),
                (self._idx(x + 1, y), self._idx(x + 1, y + 1)),
            ]
        elif direction == 'col':
            # (x,y)-(x+1,y), (x,y+1)-(x+1,y+1)
            return [
                (self._idx(x, y),     self._idx(x + 1, y)),
                (self._idx(x, y + 1), self._idx(x + 1, y + 1)),
            ]
        else:
            return None

    def _ensure_graph_bits(self):
        if self._graph_dirty:
            self._rebuild_graph_bits()

    # --------- 추가: (u-v) 엣지 몇 개를 "탐색에서만" 제거한 것처럼 보고 연결성 BFS ---------
    def _is_connected_with_removed_edges(self, removed_edges):
        # removed_edges: [(u,v), (u2,v2)] 형태, u/v는 node idx
        self._ensure_graph_bits()
        adj = self._adj_bits
        V = self._V

        # endpoint별로 제거할 이웃 비트 마스크를 만든다(딱 4개 endpoint만 영향)
        clear = {}
        for (u, v) in removed_edges:
            clear[u] = clear.get(u, 0) | (1 << v)
            clear[v] = clear.get(v, 0) | (1 << u)

        visited = 1  # start node 0 방문 (bit 0)
        stack = [0]

        while stack:
            u = stack.pop()
            nbr = adj[u] & ~visited
            cm = clear.get(u, 0)
            if cm:
                nbr &= ~cm

            while nbr:
                lsb = nbr & -nbr
                v = (lsb.bit_length() - 1)
                nbr &= (nbr - 1)
                visited |= lsb
                stack.append(v)

        return visited.bit_count() == V

     # --------- 추가: 기존 로컬 제약(겹침/교차/부분 겹침)을 빠르게 체크 ---------
    def _local_allow(self, x, y, direction):
        N = self.N
        if not (0 <= x < N - 1 and 0 <= y < N - 1):
            return False

        if direction == 'row':
            if self.row_blocks[x][y].occupied:
                return False
            # 길이 2라서 x±1 시작은 부분 겹침
            if x < N - 2 and self.row_blocks[x + 1][y].occupied:
                return False
            if x > 0 and self.row_blocks[x - 1][y].occupied:
                return False
            # 교차 금지
            if self.col_blocks[x][y].occupied:
                return False
            return True

        if direction == 'col':
            if self.col_blocks[x][y].occupied:
                return False
            # 길이 2라서 y±1 시작은 부분 겹침
            if y < N - 2 and self.col_blocks[x][y + 1].occupied:
                return False
            if y > 0 and self.col_blocks[x][y - 1].occupied:
                return False
            # 교차 금지
            if self.row_blocks[x][y].occupied:
                return False
            return True

        return False


    # 1) 로컬 제약으로 대부분 걸러낸 뒤
    # 2) 제거 엣지 2개만 반영한 연결성 BFS(비트셋)로 판정한다
    def get_place_mask(self):
        N = self.N
        n_board = (N - 1) * (N - 1)
        mask = torch.zeros(2 * n_board, dtype=torch.bool)

        # 그래프 캐시 준비(현재 상태에서 한 번만)
        self._ensure_graph_bits()

        # row 후보
        k = 0
        for x in range(N - 1):
            for y in range(N - 1):
                ok = self._local_allow(x, y, 'row')
                if ok:
                    edges = self._edges_for_wall(x, y, 'row')
                    ok = self._is_connected_with_removed_edges(edges)
                mask[k] = ok
                k += 1

        # col 후보
        k = 0
        base = n_board
        for x in range(N - 1):
            for y in range(N - 1):
                ok = self._local_allow(x, y, 'col')
                if ok:
                    edges = self._edges_for_wall(x, y, 'col')
                    ok = self._is_connected_with_removed_edges(edges)
                mask[base + k] = ok
                k += 1

        return mask.unsqueeze(0)
    
    def check_pos_allow(self,x,y):
        '''
        (x, y)가 block이 놓일 수 있는 위치인지(range안에 있는지) 확인해주는 함수
        :param x: 블럭이 놓일 위치(x값, 0base index)
        :param y: 블럭이 놓일 위치(y값, 0base index)
        :return: 없음, 놓일수 없으면 raise
        '''
        if x < 0 or x >= self.N-1 or y < 0 or y >= self.N-1:
            raise NotAllowPosException()

    def check_closed(self,x,y,direction):
        '''
        (x, y)에 direciton 방향으로 블럭이 놓일 수 있는지 확인해주는 함수
        폐공간이 있는지도 확인해줌(bfs 이용, O(n^2) 시간복잡도)
        :param x: 블럭이 놓일 위치(x값, 0base index)
        :param y: 블럭이 놓일 위치(y값, 0base index)
        :param direction: 블럭이 놓일 방향('row' or 'col') 0은 row로, 1은 col로 대응
        :return: x, y에 direction방향으로 놓일 수 있는 여부(boolean)
        '''
        if(type(direction) == int):
            if(direction == 0):
                direction = 'row'
            elif(direction == 1):
                direction = 'col'

        if direction == 'row':
            self.row_blocks[x][y].occupied = 1
        elif direction == 'col':
            self.col_blocks[x][y].occupied = 1
        start_x, start_y = 0, 0
        total_cells = self.N * self.N
        queue = deque([(start_x, start_y)])
        visited = {(start_x, start_y)}
        directions = ['w', 'a', 's', 'd']
        direction_map = {
            'w': (0, -1),
            's': (0, 1),
            'a': (-1, 0),
            'd': (1, 0)
        }
        while queue:
            x_, y_ = queue.popleft()
            for direction_ in directions:
                if self.move_allow(x_, y_, direction_):
                    dx, dy = direction_map[direction_]
                    nx, ny = x_ + dx, y_ + dy
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        if direction == 'row':
            self.row_blocks[x][y].occupied = 0
        elif direction == 'col':
            self.col_blocks[x][y].occupied = 0
        return len(visited) != total_cells


    def row_block_allow(self, x, y):
        '''
        (x, y)위치에 행 방향으로 벽돌이 놓일 수 있는지 확인해주는 함수
        :param x: 블럭이 놓일 위치(x값, 0base index)
        :param y: 블럭이 놓일 위치(y값, 0base index)
        :return: 놓일 수 있는 여부 (boolean)
        '''
        self.check_pos_allow(x,y)
        if self.row_blocks[x][y].occupied:
            return False
        if x < self.N-2 and self.row_blocks[x+1][y].occupied:
            return False
        if x > 0 and self.row_blocks[x-1][y].occupied:
            return False
        if self.col_blocks[x][y].occupied:
            return False
        return not self.check_closed(x,y,'row')

    def col_block_allow(self, x, y):
        '''
        (x, y)위치에 열 방향으로 벽돌이 놓일 수 있는지 확인해주는 함수
        :param x: 블럭이 놓일 위치(x값, 0base index)
        :param y: 블럭이 놓일 위치(y값, 0base index)
        :return: 놓일 수 있는 여부 (boolean)
        '''
        self.check_pos_allow(x,y)
        if self.col_blocks[x][y].occupied:
            return False
        if y < self.N-2 and self.col_blocks[x][y+1].occupied:
            return False
        if y > 0 and self.col_blocks[x][y-1].occupied:
            return False
        if self.row_blocks[x][y].occupied:
            return False
        return not self.check_closed(x,y,'col')

    def move_allow_w(self, x, y):
        if(x < 0 or x >= self.N or y <= 0 or y >= self.N):
            return False
        if(x < self.N-1 and self.row_blocks[x][y-1].occupied):
            return False
        if(x > 0 and self.row_blocks[x-1][y-1].occupied):
            return False
        return True
    def move_allow_a(self, x, y):
        if(x <= 0 or x >= self.N or y < 0 or y >= self.N):
            return False
        if(y >= 0 and self.col_blocks[x-1][y-1].occupied):
            return False
        if(y < self.N-1 and self.col_blocks[x-1][y].occupied):
            return False
        return True
    def move_allow_s(self, x, y):
        if(x < 0 or x >= self.N or y < 0 or y >= self.N-1):
            return False
        if(x > 0 and self.row_blocks[x-1][y].occupied):
            return False
        if(x < self.N-1 and self.row_blocks[x][y].occupied):
            return False
        return True
    def move_allow_d(self, x, y):
        if(x < 0 or x >= self.N-1 or y < 0 or y >= self.N):
            return False
        if(y > 0 and self.col_blocks[x][y-1].occupied):
            return False
        if(y < self.N-1 and self.col_blocks[x][y].occupied):
            return False
        return True

    def get_obs(self):
        '''
        각 블럭이 놓여있는지 여부를 반환하는 함수
        :return: [행 방향 블럭에 대한 ndarray, 열 방향 블럭에 대한 ndarray]
        '''
        row_obs = np.array([[self.row_blocks[i][j]._occupied for j in range(self.N-1)] for i in range(self.N-1)])
        col_obs = np.array([[self.col_blocks[i][j]._occupied for j in range(self.N-1)] for i in range(self.N-1)])
        return [row_obs, col_obs]

    def get_obs_specific(self, x, y, direction):
        '''
        특정 블럭이 놓여있는지 여부를 반환하는 함수
        :param x: 확인할 블럭의 위치(x값, 0base index)
        :param y: 확인할 블럭의 위치(y값, 0base index)
        :param direction: 블럭이 놓일 방향('row' or 'col'; 0 or 1도 가능 0은 row, 1은 col에 대응됨)
        :return: 놓여있는지 여부 (0 or 1)
        '''
        if type(direction) == int:
            if direction == 0:
                direction = 'row'
            elif direction == 1:
                direction = 'col'
            else:
                raise RowColException

        if(direction == 'row'):
            return self.row_blocks[x][y].occupied
        elif(direction == 'col'):
            return self.col_blocks[x][y].occupied
        raise RowColException



    def place_block(self, x, y, direction):
        '''
        블럭 착수하는 함수(occupy를 1로 바꾸는 거만 적용됨; 학습 외 용도로는 update_block_state(1)로 하기)
        :param x: 블럭이 놓일 위치(x값, 0base index)
        :param y: 블럭이 놓일 위치(y값, 0base index)
        :param direction: 블럭이 놓일 방향('row' or 'col'; 0 or 1도 가능 0은 row, 1은 col에 대응됨)
        :return: None
        '''
        if type(direction)==int:
            if direction == 0:
                direction = 'row'
            elif direction == 1:
                direction = 'col'
            else:
                raise RowColException

        if direction == 'row':
            if self.row_block_allow(x, y):
                self.row_blocks[x][y].update_state(1)
                self._mark_dirty()
                return True
            else:
                return False
            # else:
            #     raise NotAllowedToPlace
        elif direction == 'col':
            if self.col_block_allow(x, y):
                self.col_blocks[x][y].update_state(1)
                self._mark_dirty()
                return True
            else:
                return False
            # else: raise NotAllowedToPlace

    def update_block_state(self, x, y, direction, state):
        '''
        블럭상태를 업데이트하는 함수(state를 0, 1, 2로 바꾸는 거만 적용됨)
        :param x: 블럭이 놓일 위치(x값, 0base index)
        :param y: 블럭이 놓일 위치(y값, 0base index)
        :param direction: 블럭이 놓일 방향('row' or 'col'; 0 or 1도 가능 0은 row, 1은 col에 대응됨)
        :return: None
        '''
        if(state == 1):
            if(time.time()-self.last_time<=self.cool):
                return
            if self.place_block(x, y, direction):
                self.last_time = time.time()
                return block_encoder(x, y, direction, self.N)
            return None

        if type(direction)==int:
            if direction == 0:
                direction = 'row'
            elif direction == 1:
                direction = 'col'
            else:
                raise RowColException

        if direction == 'row':
            self.row_blocks[x][y].update_state(state)
        elif direction == 'col':
            self.col_blocks[x][y].update_state(state)

    def place_block_for_ai(self, n):
        '''
        N-1*N-1개의 칸과 2개의 방향에 대해 각각 0~2*(N-1)^2의 값으로 인덱싱해서 대응되는 좌표에 블럭을 설치해둠
        :param n: 인덱스
        :return: None
        '''
        x,y,direction = block_decoder(n,self.N)
        self.update_block_state(x,y,direction, 1)

    def set_gamestate(self, block_board):
        '''
        MCTS 노드에 저장된 game state로 board를 세팅하는 함수
        :param block_board: (2, N-1, N-1) shape의 interable 데이터
        :return: None
        '''
        for i in range(self.N-1):
            for j in range(self.N-1):
                self.row_blocks[i][j].occupied = block_board[0][i][j]
        for i in range(self.N-1):
            for j in range(self.N-1):
                self.col_blocks[i][j].occupied = block_board[1][i][j]
        self._mark_dirty()

    def reset(self):
        self.set_gamestate(np.zeros((2, self.N - 1, self.N - 1)))


    def blit(self, surface):
        '''
        현재 상태를 surface에 보이게 해둠
        :param surface: 표시할 surface
        :return: None
        '''
        for i in range(self.N):
            for j in range(self.N):
                surface.blit(self.square_images[i][j],
                             (WIDTH + X_OFFSET + (WIDTH + HEIGHT) * i, WIDTH + Y_OFFSET + (WIDTH + HEIGHT) * j))
        for i in range(self.N-1):
            for j in range(self.N-1):
                self.row_blocks[i][j].blit(surface)
                self.col_blocks[i][j].blit(surface)
        self.timer.blit(surface, time.time()-self.last_time, self.cool)

    def move_allow(self, x, y, direction):
        '''
        Player가 이동가능한지 확인하는 함수
        :param x: 현재 위치의 x값(0 base index, int)
        :param y: 현재 위치의 y값(0 base index, int)
        :param pos: 현재 위치((x, y) 꼴)
        :param direction: 이동 방향, 'w' or 'a' or 's' or 'd'
        :return: 이동 가능한지 여부(boolean)
        '''

        if direction == 'w':
            if y <= 0:
                return False
            return self.move_allow_w(x, y)
        if direction == 'a':
            if x <= 0:
                return False
            return self.move_allow_a(x, y)
        if direction == 's':
            if y >= self.N-1:
                return False
            return self.move_allow_s(x, y)
        if direction == 'd':
            if x >= self.N-1:
                return False
            return self.move_allow_d(x, y)

def get_move_mask(player, block_manager):
    x,y = player.get_pos()
    return torch.tensor([block_manager.move_allow_w(x, y), block_manager.move_allow_a(x, y), block_manager.move_allow_s(x, y), block_manager.move_allow_d(x, y)]).unsqueeze(0)

def get_place_mask(block_manager, N):
    n_board = (N-1)*(N-1)
    mask = torch.zeros(2*n_board, dtype=torch.bool)
    for i in range(n_board):
        x,y,d = block_decoder(i,N)
        mask[i] = block_manager.row_block_allow(x,y)
        mask[i+n_board] = block_manager.col_block_allow(x,y)
    return mask.unsqueeze(0)


class BlockManager_for_ai:
    def __init__(self, n):
        self.row_blocks = [[RowBlock_for_ai(i, j, 0) for j in range(n - 1)] for i in range(n - 1)]
        self.col_blocks = [[ColBlock_for_ai(i, j, 0) for j in range(n - 1)] for i in range(n - 1)]
        self.N = n
        self.closed_time = 0
        self._V = n * n
        self._adj_bits = [0] * self._V  # 각 노드의 이웃을 비트셋(int)으로 저장
        self._graph_dirty = True

    def _idx(self, x, y):
        return x * self.N + y

    def _mark_dirty(self):
        self._graph_dirty = True

    def _rebuild_graph_bits(self):
        N = self.N
        V = self._V
        adj = [0] * V

        def add_edge(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # 수평 이동 엣지: (x,y) <-> (x+1,y)
        # move_allow_d와 동일한 차단 규칙을 "엣지 단위"로 적용
        for y in range(N):
            for x in range(N - 1):
                blocked = False
                # col_blocks[x][y-1] 또는 col_blocks[x][y]가 있으면 막힘
                if y > 0 and self.col_blocks[x][y - 1].occupied:
                    blocked = True
                if y < N - 1 and self.col_blocks[x][y].occupied:
                    blocked = True
                if not blocked:
                    u = self._idx(x, y)
                    v = self._idx(x + 1, y)
                    add_edge(u, v)

        # 수직 이동 엣지: (x,y) <-> (x,y+1)
        # move_allow_s와 동일한 차단 규칙을 "엣지 단위"로 적용
        for y in range(N - 1):
            for x in range(N):
                blocked = False
                # row_blocks[x-1][y] 또는 row_blocks[x][y]가 있으면 막힘
                if x > 0 and self.row_blocks[x - 1][y].occupied:
                    blocked = True
                if x < N - 1 and self.row_blocks[x][y].occupied:
                    blocked = True
                if not blocked:
                    u = self._idx(x, y)
                    v = self._idx(x, y + 1)
                    add_edge(u, v)

        self._adj_bits = adj
        self._graph_dirty = False

    # 후보 벽이 제거하는 "딱 2개 엣지"의 endpoint 쌍 계산 ---------
    # 네 코드 기준 direction:
    #  - 'row' (가로 벽): 세로 이동(위/아래)을 2칸(가로로 길이 2) 막는다
    #  - 'col' (세로 벽): 가로 이동(좌/우)을 2칸(세로로 길이 2) 막는다
    def _edges_for_wall(self, x, y, direction):
        N = self.N
        if not (0 <= x < N - 1 and 0 <= y < N - 1):
            return None

        if direction == 'row':
            # (x,y)-(x,y+1), (x+1,y)-(x+1,y+1)
            return [
                (self._idx(x, y), self._idx(x, y + 1)),
                (self._idx(x + 1, y), self._idx(x + 1, y + 1)),
            ]
        elif direction == 'col':
            # (x,y)-(x+1,y), (x,y+1)-(x+1,y+1)
            return [
                (self._idx(x, y), self._idx(x + 1, y)),
                (self._idx(x, y + 1), self._idx(x + 1, y + 1)),
            ]
        else:
            return None

    def _ensure_graph_bits(self):
        if self._graph_dirty:
            self._rebuild_graph_bits()

    # --------- 추가: (u-v) 엣지 몇 개를 "탐색에서만" 제거한 것처럼 보고 연결성 BFS ---------
    def _is_connected_with_removed_edges(self, removed_edges):
        # removed_edges: [(u,v), (u2,v2)] 형태, u/v는 node idx
        self._ensure_graph_bits()
        adj = self._adj_bits
        V = self._V

        # endpoint별로 제거할 이웃 비트 마스크를 만든다(딱 4개 endpoint만 영향)
        clear = {}
        for (u, v) in removed_edges:
            clear[u] = clear.get(u, 0) | (1 << v)
            clear[v] = clear.get(v, 0) | (1 << u)

        visited = 1  # start node 0 방문 (bit 0)
        stack = [0]

        while stack:
            u = stack.pop()
            nbr = adj[u] & ~visited
            cm = clear.get(u, 0)
            if cm:
                nbr &= ~cm

            while nbr:
                lsb = nbr & -nbr
                v = (lsb.bit_length() - 1)
                nbr &= (nbr - 1)
                visited |= lsb
                stack.append(v)

        return visited.bit_count() == V

    # --------- 추가: 기존 로컬 제약(겹침/교차/부분 겹침)을 빠르게 체크 ---------
    def _local_allow(self, x, y, direction):
        N = self.N
        if not (0 <= x < N - 1 and 0 <= y < N - 1):
            return False

        if direction == 'row':
            if self.row_blocks[x][y].occupied:
                return False
            # 길이 2라서 x±1 시작은 부분 겹침
            if x < N - 2 and self.row_blocks[x + 1][y].occupied:
                return False
            if x > 0 and self.row_blocks[x - 1][y].occupied:
                return False
            # 교차 금지
            if self.col_blocks[x][y].occupied:
                return False
            return True

        if direction == 'col':
            if self.col_blocks[x][y].occupied:
                return False
            # 길이 2라서 y±1 시작은 부분 겹침
            if y < N - 2 and self.col_blocks[x][y + 1].occupied:
                return False
            if y > 0 and self.col_blocks[x][y - 1].occupied:
                return False
            # 교차 금지
            if self.row_blocks[x][y].occupied:
                return False
            return True

        return False

    # 1) 로컬 제약으로 대부분 걸러낸 뒤
    # 2) 제거 엣지 2개만 반영한 연결성 BFS(비트셋)로 판정한다
    def get_place_mask(self):
        s1 = time.time()
        N = self.N
        n_board = (N - 1) * (N - 1)
        mask = torch.zeros(2 * n_board, dtype=torch.bool)

        # 그래프 캐시 준비(현재 상태에서 한 번만)
        self._ensure_graph_bits()

        # row 후보
        k = 0
        for x in range(N - 1):
            for y in range(N - 1):
                ok = self._local_allow(x, y, 'row')
                if ok:
                    edges = self._edges_for_wall(x, y, 'row')
                    ok = self._is_connected_with_removed_edges(edges)
                mask[k] = ok
                k += 1

        # col 후보
        k = 0
        base = n_board
        for x in range(N - 1):
            for y in range(N - 1):
                ok = self._local_allow(x, y, 'col')
                if ok:
                    edges = self._edges_for_wall(x, y, 'col')
                    ok = self._is_connected_with_removed_edges(edges)
                mask[base + k] = ok
                k += 1
        e1 = time.time()
        self.closed_time += e1 - s1

        return mask.unsqueeze(0)

    def check_closed(self, x, y, direction):
        '''
        (x, y)에 direciton 방향으로 블럭이 놓일 수 있는지 확인해주는 함수
        폐공간이 있는지도 확인해줌(bfs 이용, O(n^2) 시간복잡도)
        :param x: 블럭이 놓일 위치(x값, 0base index)
        :param y: 블럭이 놓일 위치(y값, 0base index)
        :param direction: 블럭이 놓일 방향('row' or 'col') 0은 row로, 1은 col로 대응
        :return: x, y에 direction방향으로 놓일 수 있는 여부(boolean)
        '''
        if (type(direction) == int):
            if (direction == 0):
                direction = 'row'
            elif (direction == 1):
                direction = 'col'

        if direction == 'row':
            self.row_blocks[x][y].occupied = 1
        elif direction == 'col':
            self.col_blocks[x][y].occupied = 1
        start_x, start_y = 0, 0
        total_cells = self.N * self.N
        queue = deque([(start_x, start_y)])
        visited = {(start_x, start_y)}
        directions = ['w', 'a', 's', 'd']
        direction_map = {
            'w': (0, -1),
            's': (0, 1),
            'a': (-1, 0),
            'd': (1, 0)
        }
        while queue:
            x_, y_ = queue.popleft()
            for direction_ in directions:
                if move_allow((x_, y_), direction_, self):
                    dx, dy = direction_map[direction_]
                    nx, ny = x_ + dx, y_ + dy
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        if direction == 'row':
            self.row_blocks[x][y].occupied = 0
        elif direction == 'col':
            self.col_blocks[x][y].occupied = 0
        return len(visited) != total_cells

    def row_block_allow(self, x, y):
        '''
        (x, y)위치에 행 방향으로 벽돌이 놓일 수 있는지 확인해주는 함수
        :param x: 블럭이 놓일 위치(x값, 0base index)
        :param y: 블럭이 놓일 위치(y값, 0base index)
        :return: 놓일 수 있는 여부 (boolean)
        '''
        if x < 0 or x >= self.N - 1 or y < 0 or y >= self.N - 1:
            raise NotAllowPosException()
        if self.row_blocks[x][y].occupied:
            return False
        if x < self.N - 2 and self.row_blocks[x + 1][y].occupied:
            return False
        if x > 0 and self.row_blocks[x - 1][y].occupied:
            return False
        if self.col_blocks[x][y].occupied:
            return False
        closed = self.check_closed(x, y, 'row')
        return not closed

    def col_block_allow(self, x, y):
        '''
        (x, y)위치에 열 방향으로 벽돌이 놓일 수 있는지 확인해주는 함수
        :param x: 블럭이 놓일 위치(x값, 0base index)
        :param y: 블럭이 놓일 위치(y값, 0base index)
        :return: 놓일 수 있는 여부 (boolean)
        '''
        if self.col_blocks[x][y].occupied:
            return False
        if y < self.N - 2 and self.col_blocks[x][y + 1].occupied:
            return False
        if y > 0 and self.col_blocks[x][y - 1].occupied:
            return False
        if self.row_blocks[x][y].occupied:
            return False

        closed = self.check_closed(x, y, 'col')

        return not closed

    def move_allow_w(self, x, y):
        if (x < 0 or x >= self.N or y <= 0 or y >= self.N):
            return False
        if (x < self.N - 1 and self.row_blocks[x][y - 1].occupied):
            return False
        if (x > 0 and self.row_blocks[x - 1][y - 1].occupied):
            return False
        return True

    def move_allow_a(self, x, y):
        # FIX: y==0일 때 y-1 == -1 접근 방지
        if (x <= 0 or x >= self.N or y < 0 or y >= self.N):
            return False
        if (y > 0 and self.col_blocks[x - 1][y - 1].occupied):
            return False
        if (y < self.N - 1 and self.col_blocks[x - 1][y].occupied):
            return False
        return True

    def move_allow_s(self, x, y):
        if (x < 0 or x >= self.N or y < 0 or y >= self.N - 1):
            return False
        if (x > 0 and self.row_blocks[x - 1][y].occupied):
            return False
        if (x < self.N - 1 and self.row_blocks[x][y].occupied):
            return False
        return True

    def move_allow_d(self, x, y):
        if (x < 0 or x >= self.N - 1 or y < 0 or y >= self.N):
            return False
        if (y > 0 and self.col_blocks[x][y - 1].occupied):
            return False
        if (y < self.N - 1 and self.col_blocks[x][y].occupied):
            return False
        return True

    def get_obs(self):
        row_obs = np.array([[self.row_blocks[i][j].occupied for j in range(self.N - 1)] for i in range(self.N - 1)])
        col_obs = np.array([[self.col_blocks[i][j].occupied for j in range(self.N - 1)] for i in range(self.N - 1)])
        return [row_obs, col_obs]

    # 블럭 착수하는 함수, direction은 'row', 'col'
    def place_block(self, x, y, direction):
        if direction == 'row':
            if self.row_block_allow(x, y):
                self.row_blocks[x][y].occupy()
                self._mark_dirty()
            else:
                raise NotAllowedToPlace
        elif direction == 'col':
            if self.col_block_allow(x, y):
                self.col_blocks[x][y].occupy()
                self._mark_dirty()
            else:
                raise NotAllowedToPlace

    def place_block_for_ai(self, n):
        x, y, direction = block_decoder(n, self.N)
        self.place_block(x, y, direction)

    # MCTS 탐색의 성능을 위해 여기서만 특별히 set_occupied를 쓰지 않고 직접 occupied 변경
    # MCTS 노드에 저장된 game state로 board를 세팅
    def set_gamestate(self, block_board):
        for i in range(self.N - 1):
            for j in range(self.N - 1):
                self.row_blocks[i][j].occupied = block_board[0][i][j]
        for i in range(self.N - 1):
            for j in range(self.N - 1):
                self.col_blocks[i][j].occupied = block_board[1][i][j]
        self._mark_dirty()

    def reset(self):
        self.set_gamestate(np.zeros((2, self.N - 1, self.N - 1)))


def move_allow(pos, direction, block_manager):
    x = pos[0]
    y = pos[1]
    if direction == 'w':
        if y <= 0:
            return False
        return block_manager.move_allow_w(x, y)
    if direction == 'a':
        if x <= 0:
            return False
        return block_manager.move_allow_a(x, y)
    if direction == 's':
        if y >= block_manager.N - 1:
            return False
        return block_manager.move_allow_s(x, y)
    if direction == 'd':
        if x >= block_manager.N - 1:
            return False
        return block_manager.move_allow_d(x, y)


def place_allow(x, y, direction, block_manager):
    '''
    block을 놓을 수 있는지 확인해주는 함수
    :param x: int
    :param y: int
    :param direction: 이동 방향, 'row' or 'col'
    :param block_manager: 현재 판을 관리하고 있는 BlockManager
    :return: 이동 가능한지 여부(boolean)
    '''
    if direction == 'row':
        return block_manager.row_block_allow(x, y)
    elif direction == 'col':
        return block_manager.col_block_allow(x, y)
    raise RowColException