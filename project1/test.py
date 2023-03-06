import numpy as np
import time
import queue
board_weight = [[1, 8, 3, 7, 7, 3, 8, 1], [8, 3, 2, 5, 5, 2, 4, 8], [3, 2, 6, 6, 6, 6, 2, 3], [7, 5, 6, 4, 4, 6, 5, 7], [
    7, 5, 6, 4, 4, 6, 5, 7], [3, 2, 6, 6, 6, 6, 2, 3], [8, 3, 2, 5, 5, 2, 4, 8], [1, 8, 3, 7, 7, 3, 8, 1]]


def find_candidate(disc, chessboard):
    candidate = []
    directions = ([1, 0], [-1, 0], [0, 1], [0, -1], [-1, -1], [1, 1],
                  [-1, 1], [1, -1])  # ↓ → ↑ ← ↖ ↗ ↙ ↘
    blank = filter(lambda b: 0 <= b[0] < 8 and 0 <=
                   b[1] < 8 and chessboard[b] == 0, map(lambda d: (disc[0] + d[0], disc[1] + d[1]), directions))
    for i in list(blank):
        x = disc[0]
        y = disc[1]
        dir = (x - i[0], y - i[1])
        while True:
            x += dir[0]
            y += dir[1]
            if 0 <= x < 8 and 0 <= y < 8:
                if chessboard[x, y] == -1:
                    candidate.append(i)
                    break
                elif chessboard[x, y] == 0:
                    break
                else:
                    continue
            else:
                break
    return candidate


chessboard = np.array([[1, 1, -1, -1, -1, -1, -1, 0], [1, 1, 1, -1, -1, -1, -1, -1], [1, 1, 1, -1, 1, 1, -1, -1], [0, 1, -1, -1,
                      1, -1, -1, -1], [1, 1, -1, -1, -1, -1, 1, -1], [0, 1, -1, -1, 1, 1, 1, -1], [0, -1, 1, 1, 1, 0, 1, -1], [0, -1, 1, 0, 0, 1, 1, 1]])
candidate_list = []
result = queue.PriorityQueue()
start = time.time()
opo = np.where(chessboard == 1)
opo = list(zip(opo[0], opo[1]))
for i in opo:
    candidate_list += find_candidate(
        i, chessboard)
for j in set(candidate_list):
    result.put((-board_weight[j[0]][j[1]], j))
print(result.get()[1])
run_time = (time.time() - start)
print(run_time)
a = []
b = []
print(set(a))
b += set(a)
print(b)
