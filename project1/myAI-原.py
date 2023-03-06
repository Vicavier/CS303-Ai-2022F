
from typing import DefaultDict
import numpy as np
import random
import time
import queue

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
board_weight = [[1, 8, 3, 7, 7, 3, 8, 1], [8, 3, 2, 5, 5, 2, 4, 8], [3, 2, 6, 6, 6, 6, 2, 3], [7, 5, 6, 4, 4, 6, 5, 7], [
    7, 5, 6, 4, 4, 6, 5, 7], [3, 2, 6, 6, 6, 6, 2, 3], [8, 3, 2, 5, 5, 2, 4, 8], [1, 8, 3, 7, 7, 3, 8, 1]]


class Board(DefaultDict):
    """A board has the player to move, a cached utility value, 
    and a dict of {(x, y): player} entries, where player is '-1' or '1'."""

    def __init__(self, chessboard_size=8, to_move=None, **kwds):
        self.__dict__.update(
            chessboard_size=chessboard_size, to_move=to_move, **kwds)

    def new(self, changes: dict, **kwds) -> 'Board':
        "Given a dict of {(x, y): contents} changes, return a new Board with the changes."
        board1 = Board(chessboard_size=self.chessboard,
                       **kwds)  # new一个新的board1
        board1.update(self)  # 用调用new方法的board（self）更新board1
        board1.update(changes)  # update board1中的字典
        return board1


class AI(object):
    # chessboard_size,color,time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        # 初始化棋盘
        self.initial = Board(chessboard_size=chessboard_size,
                             to_move=color, utility=0)

    def actions(self, board):
        """Legal moves are any square not yet taken."""
        return self.squares - set(board)

    def find_candidate(self, disc, chessboard):
        candidate = []
        # 找到该对方妻子周边所有的空白格子
        directions = ([1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1], [1, 1],
                      [-1, 1], [1, -1])  # ↓ ↑ → ← ↖ ↘ ↗ ↙
        blank = filter(lambda b: 0 <= b[0] < self.chessboard_size and 0 <=
                       b[1] < self.chessboard_size and chessboard[b] == COLOR_NONE, map(lambda d: (disc[0] + d[0], disc[1] + d[1]), directions))
        # 反方向查找有无己方棋子，有则该空格有效
        for i in list(blank):
            x = disc[0]
            y = disc[1]
            dir = (x - i[0], y - i[1])
            while True:
                x += dir[0]
                y += dir[1]
                if 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size:
                    if chessboard[x, y] == self.color:
                        candidate.append(i)
                        break
                    elif chessboard[x, y] == COLOR_NONE:
                        break
                    else:
                        continue
                else:
                    break
        return candidate

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # find all oponent's disc
        opo = np.where(chessboard == -self.color)
        opo = list(zip(opo[0], opo[1]))
        candidate = []
        for i in opo:
            candidate += self.find_candidate(
                i, chessboard)
        self.candidate_list = list(set(candidate))
