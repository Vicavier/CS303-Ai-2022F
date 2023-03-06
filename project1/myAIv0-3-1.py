
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
board_weight = np.array([[1, 8, 3, 7, 7, 3, 8, 1], [8, 3, 2, 5, 5, 2, 4, 8], [3, 2, 6, 6, 6, 6, 2, 3], [7, 5, 6, 4, 4, 6, 5, 7], [
    7, 5, 6, 4, 4, 6, 5, 7], [3, 2, 6, 6, 6, 6, 2, 3], [8, 3, 2, 5, 5, 2, 4, 8], [1, 8, 3, 7, 7, 3, 8, 1]])
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
        self.squares = {(x, y) for x in range(chessboard_size)
                        for y in range(chessboard_size)}
        self.initial = Board(chessboard_size, color, utility=0)

    def actions(self, board):
        """Legal moves are any square not yet taken."""
        action = []
        player = board.to_move
        # 找到该对方棋子
        opo = np.where(chessboard == -player)
        opo = list(zip(opo[0], opo[1]))
        # 找到该对方妻子周边所有的空白格子
        directions = ([1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1], [1, 1],
                      [-1, 1], [1, -1])  # ↓ ↑ → ← ↖ ↘ ↗ ↙
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
                        if chessboard[x, y] == player:
                            action.append(disc)
                            break
                        elif chessboard[x, y] == COLOR_NONE:
                            break
                        else:
                            continue
                    else:
                        break
        return action

    def result(self, chessboard, player, square):
        """Place a marker for current player on square."""
        chessboard[square[0], square[1]] = player
        print(chessboard)
        win = self.is_win(player, square)
        '''
        if not win:
            utility = 0
        else:
            if player == 'X':
                utility = 1
            else:
                utility = -1
        '''
        self.utility = (board_weight[square[0]][square[1]] if not win else +10086 if player ==
                        '-1' else -1)  # 三目运算符中套了一个三目运算符

    def utility(self, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return self.utility if player == '-1' else -self.utility

    def minimax_search(self, chessboard):
        """Search game tree to determine best move; return (value, move) pair."""
        d = 0

        def max_value(chessboard, player, deep):
            deep += 1
            if deep == 4:
                return self.utility, None
            if self.is_terminal(player):
                return self.utility, None
            v, move = -infinity, None
            for a in self.actions(chessboard, player):
                self.result(chessboard, player, a)
                v2, _ = min_value(chessboard, -player, deep)
                if v2 > v:
                    v, move = v2, a
            return v, move

        def min_value(chessboard, player, deep):
            deep += 1
            if deep == 4:
                return self.utility, None
            if self.is_terminal(player):
                return
            v, move = infinity, None
            for a in self.actions(chessboard, player):
                self.result(chessboard, player, a)
                v2, _ = max_value(chessboard, player, deep)
                if v2 < v:
                    v, move = v2, a
            return v, move

        self.candidate_list.append(max_value(chessboard, self.color, d)[1])

    def is_terminal(self, player):
        """A board is a terminal state if it is won or there are no empty squares."""
        return False

    def is_win(self, player, square):
        """True if player has k pieces in a line through square."""
        return False

    def go(self, chessboard):
        # Clear candidate_list and reset chessboard, must do this step
        self.candidate_list.clear()
        # find all oponent's disc
        print(chessboard)
        self.minimax_search(chessboard)
        self.candidate_list = self.actions()


Ai = AI(5, -1, 5)
dict = {(0, 0): '1', (0, 1): '-1'}
print(Ai.initial.new(dict))
print(set(Ai.initial.new(dict)))

chessboard = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0], [
    0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
# print(board_weight)
# Ai.go(chessboard)
