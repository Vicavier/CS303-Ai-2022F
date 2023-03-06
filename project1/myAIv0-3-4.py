
from typing import DefaultDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
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
# (1, 0), (0, 6), (7, 1), (6, 7)
board_weight = np.array([[-20, 4, -5, -4, -4, -5, 20, -20],
                         [20, -5, -5, 5, 5, -5, -5, 4],
                         [-5, -5, 7, 7, 7, 7, -5, -5],
                         [-4, 5, 7, 6, 6, 7, 5, -4],
                         [-4, 5, 7, 6, 6, 7, 5, -4],
                         [-5, -5, 7, 7, 7, 7, -5, -5],
                         [4, -5, -5, 5, 5, -5, -5, 20],
                         [-20, 20, -5, -4, -4, -5, 4, -20]])
# board_weight = np.array([[-20, 20, 1, -4, -4, 1, 20, -20],
#                          [20, 2, 2, 4, 4, 2, 2, 20],
#                          [1, 2, 7, 7, 7, 7, 2, 1],
#                          [-4, 4, 7, 6, 6, 7, 4, -4],
#                          [-4, 4, 7, 6, 6, 7, 4, -4],
#                          [1, 2, 7, 7, 7, 6, 3, 2],
#                          [20, 2, 2, 4, 4, 2, 2, 20],
#                          [-20, 20, 1, -4, -4, 1, 20, -20]])
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

    def __init__(self, chessboard_size=8, color=-1, time_out=5, deep=5, method=0):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        self.deep = deep
        self.method = method
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

        # 当前玩家落下square后，更新棋盘

        def update(board1):
            add = 0
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
                            # add += len(temp)
                            for i in temp.keys():
                                add += 0.1 * board_weight[i[0]][i[1]]
                            break
                        else:
                            temp[(x, y)] = player
                            continue
                    else:
                        break
            return add
        add = update(board1)
        win = self.is_win(board1, player)  # 判断当前玩家是否获胜
        actions = self.actions(board1)
        a = board_weight[square[0]][square[1]]
        b = len(actions)
        # 前期看重棋盘权重的同时更看重每步增加的个数
        if len(board1) <= 15:
            board1.utility = ((0.5*a - 2*add) if not win else +20086 if player ==
                              self.color else -10086)
        # 中期更看重棋盘权重，因为涉及到抢占稳定点
        elif 15 < len(board1) <= 35:
            board1.utility = ((a - 1.5 * add) if not win else +20086 if player ==
                              self.color else -10086)
        # 后期
        elif 35 < len(board1) < 50:
            board1.utility = ((a - 0.6 * b - 0.5 * add) if not win else +20086 if player ==
                              self.color else -10086)
        else:
            board1.utility = ((0.5*a - b - 0.5 * add) if not win else +20086 if player ==
                              self.color else -10086)
        return board1

    def utility(self, board, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return board.utility if player == self.color else -board.utility

    def alphabeta_search(self, state):
        """Search game to determine best action; use alpha-beta pruning.
        As in [Figure 5.7], this version searches all the way to the leaves."""
        d = 0
        T1 = time.time()
        player = state.to_move

        def max_value(state, deep, alpha, beta):
            deep += 1
            if deep == self.deep:
                return state.utility, None
            if self.is_terminal(state):
                return self.utility(state, player), None
            v, move = -infinity, None
            actions = self.actions(state)
            for a in actions:
                # T2 = time.time()
                # if (T2-T1) > 4.98:
                #     return v, move
                if a in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                    continue
                if a in [(1, 0), (0, 6), (7, 1), (6, 7)]:
                    return 10086, a
                v2, _ = min_value(self.result(state, a), deep, alpha, beta)
                # if deep == 1:
                #     print('move:(%d,%d),u:%.1f' % (a[0], a[1], v2))
                if v2 > v:
                    v, move = v2, a
                elif move and v2 == v:
                    if board_weight[a[0]][a[1]] > board_weight[move[0]][move[1]]:
                        v, move = v2, a
                    # print('layer: %d, move: (%d,%d), u: %.1f' %
                    #       (deep, move[0], move[1], v))

                if v >= beta:
                    break
                if v > alpha:
                    alpha = v
            return v, move

        def min_value(state, deep, alpha, beta):
            deep += 1
            if deep == self.deep:
                return state.utility, None
            if self.is_terminal(state):
                return self.utility(state, player), None
            v, move = infinity, None
            actions = self.actions(state)
            for a in actions:
                if a in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                    continue
                if a in [(0, 1), (6, 0), (7, 6), (1, 7)]:
                    return 10086, a
                v2, _ = max_value(self.result(state, a), deep, alpha, beta)
                if v2 < v:
                    v, move = v2, a
                    # print('layer: %d, move: (%d,%d), u: %.1f' %
                    #       (deep, move[0], move[1], v))
                if v <= alpha:
                    break
                if v < beta:
                    beta = v
            return v, move

        candidate = max_value(state, d, -infinity, infinity)[1]
        if candidate:
            self.candidate_list.append(candidate)
        # T2 = time.time()
        # print(T2-T1)

    def minimax_search(self, state):
        """Search game tree to determine best move; return (value, move) pair."""
        d = 0
        T1 = time.time()

        def max_value(state, deep):
            deep += 1
            if deep == self.deep:
                return state.utility, None
            if self.is_terminal(state):
                return self.utility(state, player), None
            v, move = -infinity, None
            actions = self.actions(state)
            for a in actions:
                # T2 = time.time()
                # if (T2-T1) > 4.99:
                #     return v, move
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
            if deep == self.deep:
                return state.utility, None
            if self.is_terminal(state):
                return self.utility(state, player), None
            v, move = infinity, None
            actions = self.actions(state)
            for a in actions:
                # T2 = time.time()
                # if (T2-T1) > 4.99:
                #     return v, move
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
        if self.method == 0:
            self.alphabeta_search(self.initial)
        else:
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


cost = [[0.0], [0.0], [0.0], [0.0]]
for d in (5, 6, 7, 8):
    Ai0 = AI(8, -1, 5, d, 0)
    Ai1 = AI(8, 1, 5, d, 0)
    coboard = Board(8, to_move=None, utility=0)
    coboard.update({(3, 3): -1, (3, 4): 1, (4, 3): 1, (4, 4): -1})

    board = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0], [
        0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])

    cnt = 0

    while len(coboard) != 60:
        if cnt % 2 == 0:
            coboard.to_move = -1
            t1 = time.time()
            Ai0.go(board)
            t2 = time.time()
            cost[d-5].append(round(t2-t1, 2))
            print('%d:player1 move:(%d,%d)' %
                  (cnt, Ai0.candidate_list[-1][0], Ai0.candidate_list[-1][1]))
            coboard = Ai0.result(coboard, Ai0.candidate_list[-1])
            for k, v in coboard.items():
                board[k] = v
            # print(board)
            cnt += 1
        else:
            coboard.to_move = 1
            t1 = time.time()
            Ai1.go(board)
            t2 = time.time()
            cost[d-5].append(round(t2-t1, 2))
            print('%d:player2 move:(%d,%d)' %
                  (cnt, Ai1.candidate_list[-1][0], Ai1.candidate_list[-1][1]))
            coboard = Ai1.result(coboard, Ai1.candidate_list[-1])
            for k, v in coboard.items():
                board[k] = v
            # print(board)
            cnt += 1

    print(cost[d-5][1:])
    plt.plot(cost[d-5][1:])

plt.xlabel('move')
plt.ylabel('time/s')
plt.legend(['deep=5', 'deep=6', 'deep=7', 'deep=8'])
y_major_locator = MultipleLocator(10)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.show()
