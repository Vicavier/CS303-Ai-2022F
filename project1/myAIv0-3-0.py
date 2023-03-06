
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
board_weight = [[1, 8, 3, 7, 7, 3, 8, 1], [8, 3, 2, 5, 5, 2, 4, 8], [3, 2, 6, 6, 6, 6, 2, 3], [7, 5, 6, 4, 4, 6, 5, 7], [
    7, 5, 6, 4, 4, 6, 5, 7], [3, 2, 6, 6, 6, 6, 2, 3], [8, 3, 2, 5, 5, 2, 4, 8], [1, 8, 3, 7, 7, 3, 8, 1]]
infinity = math.inf


class State(DefaultDict):
    """A board has the player to move, a cached utility value, 
    and a dict of {(x, y): color} entries, where player is '-1' or '1'."""
    empty = '0'
    off = '#'

    def __init__(self, chessboard_size=8, to_move=None, **kwds):
        self.__dict__.update(chessboard_size=chessboard_size,
                             to_move=to_move, **kwds)

    def new(self, changes: dict, **kwds) -> 'State':
        "Given a dict of {(x, y): contents} changes, return a new Board with the changes."
        board1 = State(self.chessboard_size, **kwds)  # new一个新的board1
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

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        # 初始化棋盘
        self.chessboard = []
        self.initial = State(chessboard_size=chessboard_size,
                             to_move=color, utility=0)

    def actions(self, chessboard):
        """Legal moves are any square not yet taken."""
        action = []
        # 找到该对方棋子
        opo = np.where(chessboard == -self.color)
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
                        if chessboard[x, y] == self.color:
                            action.append(disc)
                            break
                        elif chessboard[x, y] == COLOR_NONE:
                            break
                        else:
                            continue
                    else:
                        break
        return action

    def result(self, state, move):
        """Place a marker for current player on square."""

        state = state.new({move: self.color}, to_move=(
            '-1' if self.color == '-1' else '1'))
        # win = k_in_row(board, player, square, self.k)
        '''
        if not win:
            utility = 0
        else:
            if player == 'X':
                utility = 1
            else:
                utility = -1
        '''
        # state.utility = (0 if not win else +1 if player ==
        #                  'X' else -1)  # 三目运算符中套了一个三目运算符
        return state

    def utility(self):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return self.state.utility if self.color == '-1' else -self.state.utility

    def minimax_search(self, state):
        """Search game tree to determine best move; return (value, move) pair."""

        player = state.to_move

        def max_value(state):
            if self.is_terminal(state):
                return self.utility(state, player), None
            v, move = -infinity, None
            for a in self.actions(state):
                v2, _ = min_value(self.result(state, a))
                if v2 > v:
                    v, move = v2, a
            return v, move

        def min_value(state):
            # TODO: implement function min_value
            if self.is_terminal(state):
                return self.utility(state, player), None
            v, move = infinity, None
            for a in self.actions(state):
                v2, _ = max_value(self.result(state, a))
                if v2 < v:
                    v, move = v2, a
            return v, move

        return max_value(state)

    def go(self, chessboard):
        # Clear candidate_list and reset chessboard, must do this step
        self.chessboard = chessboard
        self.candidate_list.clear()
        # find all oponent's disc

        print(self.chessboard)
        print(self.actions(self.chessboard))
        self.initial = self.result(self.initial, (0, 1))
        print(self.initial)


Ai = AI(8, -1, 5)
chessboard = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0], [
    0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
Ai.go(chessboard)
