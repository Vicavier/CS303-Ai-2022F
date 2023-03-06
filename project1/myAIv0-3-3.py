
from typing import DefaultDict
import numpy as np
import random
import math
import time
import queue

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
directions = ([1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1], [1, 1],
              [-1, 1], [1, -1])  # ↓ ↑ → ← ↖ ↘ ↗ ↙
board_weight = np.array([[-10, 100, -10,  -10,  -10,  -10,  100, -10],
                         [100, 3, -10,  7,  7, -10, 3,   100],
                         [-10, -10,   10, 10, 10, 10, -10,  -10],
                         [-10,   7,   10, 4,  4,  10, 7,   -10],
                         [-10,   7,   10, 4,  4,  10, 7,   -10],
                         [-10,   -10,   10, 10, 10, 10, -10, -10],
                         [100, 3,   -10,  7,  7,  -10,  3, 100],
                         [-10, 100, -10,  -10,  -10,  -10, 100, -10]])
infinity = math.inf

# Board类继承了DefaultDict类，本质上Board是一个字典，初始化时，会自动生成棋盘字典（长度为0，此时为空字典）
# 打印时，若键值missing则自动填充0
# 后续通过new方法增加或更新字典中的值


class Board(DefaultDict):
    """A board has the player to move, a cached utility value, 
    and a dict of {(x, y): player} entries, where player is 'X' or 'O'."""
    empty = '0'
    off = '#'

    def __init__(self, chessboard_size=8, to_move=None, **kwds):
        self.__dict__.update(
            chessboard_size=chessboard_size, to_move=to_move, **kwds)

    def new(self, changes: dict, **kwds) -> 'Board':
        "Given a dict of {(x, y): contents} changes, return a new Board with the changes."
        board1 = Board(chessboard_size=self.chessboard_size,
                       **kwds)  # new一个新的board1
        board1.update(self)  # 用调用new方法的board（self）更新board1

        board1.update(changes)  # update board1中的字典
        return board1

    def __missing__(self, loc):
        x, y = loc
        if 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size:
            return self.empty
        else:
            return self.off

    def __hash__(self):
        return hash(tuple(sorted(self.items()))) + hash(self.to_move)

    def __repr__(self):
        def row(y): return ' '.join(self[x, y]
                                    for x in range(self.chessboard_size))
        return '\n'.join(map(row, range(self.chessboard_size))) + '\n'


class AI(object):
    '''chessboard_size,color,time_out passed from agent'''

    def __init__(self, chessboard_size=8, color=-1, time_out=5):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        self.squares = {(x, y) for x in range(chessboard_size)
                        for y in range(chessboard_size)}
        self.initial = Board(chessboard_size=chessboard_size,
                             to_move=color, utility=0)

    def actions(self, board):
        """Legal moves are any square not yet taken."""
        action = []
        opo = []
        player = board.to_move
        chessboard = np.zeros(
            (self.chessboard_size, self.chessboard_size), dtype=int)
        # 找到对方棋子
        for key in board.keys():
            chessboard[key] = board[key]
            if board[key] == -player:
                opo.append(key)
        # 找到该对方妻子周边所有的空白格子
        for i in opo:
            neighbor = list(filter(lambda b: 0 <= b[0] < self.chessboard_size and 0 <=
                                   b[1] < self.chessboard_size and chessboard[b] == COLOR_NONE, map(lambda d: (i[0] + d[0], i[1] + d[1]), directions)))

        # 反方向查找有无己方棋子，有则该空格有效
            for disc in neighbor:
                x = i[0]
                y = i[1]
                dir = (x - disc[0], y - disc[1])
                while True:
                    x += dir[0]
                    y += dir[1]
                    if 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size:
                        if chessboard[x][y] == player:
                            action.append(disc)
                            break
                        elif chessboard[x, y] == COLOR_NONE:
                            break
                        else:
                            continue
                    else:
                        break
        return set(action)

    def result(self, board, square):
        """Place a marker for current player on square."""
        player = board.to_move  # 当前玩家
        board1 = board.new({square: player}, to_move=(
            1 if player == -1 else -1))  # board1的玩家已置换为对手
        add = 0
        # 当前玩家落下square后，更新棋盘
        for dir in directions:
            x = square[0]
            y = square[1]
            temp = {}
            while True:
                x += dir[0]
                y += dir[1]

                if 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size:
                    if board1[(x, y)] == player:
                        board1.update(temp)
                        add += len(temp)
                        break
                    elif board1[(x, y)] == -player:
                        break
                    else:
                        continue
                else:
                    break
        win = self.is_win(board1, player)  # 判断当前玩家是否获胜
        actions = self.actions(board1)
        a = board_weight[square[0]][square[1]]
        b = len(actions)
        if len(board1) <= 15:
            board1.utility = ((0.5*a - 1.5*add) if not win else +10086 if player ==
                              self.color else -10086)  # 三目运算符中套了一个三目运算符
        elif 15 < len(board1) <= 35:
            board1.utility = ((0.5*a - add) if not win else +10086 if player ==
                              self.color else -10086)  # 三目运算符中套了一个三目运算符
        elif 35 < len(board1) < 50:
            board1.utility = ((0.5*a + 0.6 * b - 0.5 * add) if not win else +10086 if player ==
                              self.color else -10086)  # 三目运算符中套了一个三目运算符
        else:
            board1.utility = ((0.5*a - b - 0.5 * add) if not win else +10086 if player ==
                              self.color else -10086)  # 三目运算符中套了一个三目运算符
        return board1

    def utility(self, board, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return board.utility if player == self.color else -board.utility

    def minimax_search(self, state):
        """Search game tree to determine best move; return (value, move) pair."""
        d = 0
        T1 = time.time()

        def max_value(state, deep):
            deep += 1
            if deep == 4:
                return state.utility, None
            if self.is_terminal(state):
                return self.utility(state, player), None
            v, move = -infinity, None
            actions = self.actions(state)
            for a in actions:
                T2 = time.time()
                if (T2-T1) > 4.99:
                    return v, move
                if a in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                    continue
                if a in [(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (7, 6), (6, 7)]:
                    return 10086, a
                v2, _ = min_value(self.result(state, a), deep)
                if v2 >= v:
                    v, move = v2, a
                    # print('layer: %d, move: (%d,%d), u: %.1f' %
                    #       (deep, move[0], move[1], v))
            return v, move

        def min_value(state, deep):
            deep += 1
            if deep == 4:
                return state.utility, None
            if self.is_terminal(state):
                return self.utility(state, player), None
            v, move = infinity, None
            actions = self.actions(state)
            for a in actions:
                T2 = time.time()
                if (T2-T1) > 4.99:
                    return v, move
                # if a in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                #     return -10086, a
                # if a in [(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (7, 6), (6, 7)]:
                #     return -10086, a
                v2, _ = max_value(self.result(state, a), deep)
                if v2 <= v:

                    v, move = v2, a
                    # print('layer: %d, move: (%d,%d), u: %.1f' %
                    #       (deep, move[0], move[1], v))
            return v, move

        player = state.to_move
        candidate = max_value(state, d)[1]
        if candidate:
            self.candidate_list.append(candidate)

    def is_terminal(self, board):
        """A board is a terminal state if it is won or there are no empty squares."""
        return len(board) == self.chessboard_size * self.chessboard_size

    def go(self, chessboard):
        # Clear candidate_list and reset chessboard, must do this step
        self.candidate_list.clear()
        # find all oponent's disc
        opo = np.where(chessboard != COLOR_NONE)
        opo = list(zip(opo[0], opo[1]))

        board = {(x, y): chessboard[x][y] for x, y in opo}
        self.initial.update(board)
        self.candidate_list += self.actions(self.initial)
        self.minimax_search(self.initial)

    def is_win(self, board, player):
        """True if player has k pieces in a line through square."""
        if len(board) < self.chessboard_size * self.chessboard_size:
            return False
        else:
            sum = 0
            for i in board.values():
                sum += i
            if sum <= 0 and player == -1:
                return True
            elif sum >= 0 and player == 1:
                return True
            else:
                return False


# Ai = AI(8, -1, 5)
# # dict = {(0, 0): '1', (0, 1): '-1'}
# # print(Ai.initial.new(dict))
# # print(set(Ai.initial.new(dict)))

# # chessboard1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0], [
# #     0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
# # chessboard2 = np.array([[0, 1, 0, 0, 0, 0, 0, -1], [0, 0, 1, -1, -1, 0, 0, -1], [0, -1, 0, 1, -1, -1, -1, -1], [0, 0, -1, 1, -
# #                        1, -1, 1, 1], [0, 0, -1, -1, 1, -1, 1, 1], [0, 0, 0, -1, -1, 1, 1, 1], [0, 0, 0, 0, -1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
# # chessboard3 = np.array([[1, 1,y  1, -1,  0, 0,  0,  0], [1,  1,  1, -1, 0, 0,  0,  0], [1,  1, -1, 0,  0, 0, 0, 0], [-1, -1, 0, 0,
# #                        0, 0, 0, 0], [0, 0,  0, 0, 0,  0, 0, 0], [0,  0, 0, 0, 0, 0,  0, 0], [0,  0,  0,  0,  0, 0,  0, 0], [0, 0,  0,  0, 0, 0, 0, 0]])
# # chessboard4 = np.array([[0, 0, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, -1], [0, -1, -1, -1, -1,
# #                        0, -1, 0], [0, 0, -1, -1, 1, -1, 0, 0], [0, 0, 0, -1, 1, -1, -1, 0], [0, 0, -1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]])
# chessboard5 = np.array([[0, 0, 0, 0, -1, 0, 1, 0], [0, 0, 0, -1, 1, -1, 0, 1], [0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, -1,
#                        1, 1, 0, 0], [0, 1, 1, -1, 1, 1, 0, 0], [0, 1, 0, -1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
# chessboard6 = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, -1, 1, -1, 0, 0], [0, 0, 0, 1, -1, -1, 0, 0],
#                        [0, 0, 1, -1, 1, -1, 0, 1], [0, 1, 1, 1, 1, 1, -1, 0], [0, 0, 0, 0, 1, 1, 1, -1], [0, 0, 0, 0, 0, 0, 0, 0]])
# print(board_weight)
# # print(chessboard4)
# print(chessboard5)
# Ai.go(chessboard5)
# # print(board_weight)
# print(Ai.candidate_list)
