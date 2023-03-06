
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

    def actions(self, chessboard, player):
        """Legal moves are any square not yet taken."""
        action = []
        opo = np.where(chessboard == -player)
        opo = list(zip(opo[0], opo[1]))
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
                        elif chessboard[x][y] == COLOR_NONE:
                            break
                        else:
                            continue
                    else:
                        break
        return set(action)


Ai = AI(8, -1, 5)
chessboard0 = np.array([[1, 1, -1, -1, -1, -1, -1, 0], [1, 1, 1, -1, -1, -1, -1, -1], [1, 1, 1, -1, 1, 1, -1, -1], [0, 1, -1, -1,
                       1, -1, -1, -1], [1, 1, -1, -1, -1, -1, 1, -1], [0, 1, -1, -1, 1, 1, 1, -1], [0, -1, 1, 1, 1, 0, 1, -1], [0, -1, 1, 0, 0, 1, 1, 1]])
# print(chessboard6)
# print(board_weight)
# Ai.go(chessboard6)
# for cb in chessboard6:
#     Ai.actions()
# print(chessboard0)
player = -1
cnt = 0
for cb in chessboard6:
    temp = len(Ai.actions(cb, player))
    if temp > cnt:
        cnt = temp
    player *= -1
print(cnt)
# print(Ai.candidate_list)
