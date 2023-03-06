import random
import time
import argparse
import multiprocessing as mp
import copy
import numpy as np

N = 9999
VERTICES = 0
DEPOT = 0
REQUIRED_EDGES = 0
CAPACITY = 0

Required = []
child = []
shortestPath = []
map = []
TIME = 0
SEED = 0
filepath = ''


def readFile(filepath):
    global VERTICES, DEPOT, REQUIRED_EDGES, CAPACITY, Required, child
    specification = []
    with open(filepath, encoding='utf-8') as f:
        for _ in range(8):
            specification.append(f.readline().split(' : ')[1].strip())

        VERTICES = int(specification[1])
        DEPOT = int(specification[2])
        REQUIRED_EDGES = int(specification[3])
        CAPACITY = int(specification[6])

        map = [[(N, 0) for _ in range(VERTICES+1)] for _ in range(VERTICES+1)]
        for _ in range(VERTICES):
            child.append([])
        index = 0
        while True:
            line = f.readline()
            if line == 'END':
                break
            elif line == 'NODES       COST         DEMAND\n':
                continue
            else:
                data = line.split()
                a = int(data[0])
                b = int(data[1])
                cost = int(data[2])
                demand = int(data[3])
                map[a][b] = (cost, demand)
                map[b][a] = (cost, demand)
                child[a - 1].append((b, cost))
                child[b - 1].append((a, cost))
                if demand != 0:
                    Required.append((a, b))
                    index += 1
    return map


def dijkstra(child: list):
    n = len(child)
    # father = np.zeros((n, n), dtype=int)
    shortestPath = np.zeros((n + 2, n + 2), dtype=int)
    for i in range(1, 1 + n):
        visited = [0] * (n + 2)
        dis = [N] * (n + 2)
        # f = [0] * n
        visited[i] = 1
        dis[i] = 0
        heap = []
        heap.append(i)
        while len(heap) != 0:
            index = 0
            min = N
            for _ in range(len(heap)):
                if dis[heap[_]] < min:
                    index = _
                    min = dis[heap[_]]

            cur = heap[index]
            heap.pop(index)
            for j in child[cur - 1]:
                if visited[j[0]] == 0:
                    visited[j[0]] = 1
                    dis[j[0]] = dis[cur] + j[1]
                    heap.append(j[0])
                    # f[j[0] - 1] = cur
                else:
                    if dis[j[0]] > dis[cur] + j[1]:
                        dis[j[0]] = dis[cur] + j[1]
                        # f[j[0] - 1] = cur
        shortestPath[i] = dis
        # father[i] = f
    return shortestPath


def better(arcmin, arc, load, now):
    strategy = random.random()
    if strategy < 0.15:
        if shortestPath[arcmin[1]][DEPOT] > shortestPath[arc[1]][DEPOT]:
            return True
        else:
            return False
    elif strategy < 0.3:
        if shortestPath[arcmin[1]][DEPOT] < shortestPath[arc[1]][DEPOT]:
            return True
        else:
            return False
    elif strategy < 0.6:  # demand/cost
        if map[arcmin[0]][arcmin[1]][1]/(shortestPath[now][arcmin[0]]+map[arcmin[0]][arcmin[1]][0]) > map[arc[0]][arc[1]][1]/(shortestPath[now][arc[0]]+map[arc[0]][arc[1]][0]):
            return True
        else:
            return False
    elif strategy < 0.7:
        if map[arcmin[0]][arcmin[1]][1]/(shortestPath[now][arcmin[0]]+map[arcmin[0]][arcmin[1]][0]) < map[arc[0]][arc[1]][1]/(shortestPath[now][arc[0]]+map[arc[0]][arc[1]][0]):
            return True
        else:
            return False
    elif strategy < 1:
        if load < CAPACITY / 2 and shortestPath[arcmin[1]][DEPOT] > shortestPath[arc[1]][DEPOT]:
            return True
        elif load > CAPACITY / 2 and shortestPath[arcmin[1]][DEPOT] < shortestPath[arc[1]][DEPOT]:
            return True
        else:
            return False


def pathscan(map):
    depot = DEPOT
    free = copy.deepcopy(Required)
    k = -1
    Route = []
    load = []
    cost = []
    # print(free)
    total = 0  # cost
    while len(free) != 0:
        k += 1
        Route.append(0)
        load.append(0)
        cost.append(0)
        now = depot
        Route[k] = []
        arc = (0, 0)
        while True:
            d = N
            index = -1
            q = 0
            r = random.random()
            # print(r)
            if r < 0.5 and len(Route[k]) == 0:
                if len(free) == 0:
                    break
                rr = int(random.random()*(len(free)))
                aa = free[rr]
                # print(rr,len(free)-1)
                if load[k] + map[aa[0]][aa[1]][1] <= CAPACITY:
                    if shortestPath[now][aa[0]] <= shortestPath[now][aa[1]]:
                        dmin = shortestPath[now][aa[0]]
                        arcmin = (aa[0], aa[1])
                    else:
                        dmin = shortestPath[now][aa[1]]
                        arcmin = (aa[1], aa[0])
                    d = dmin
                    q = map[aa[0]][aa[1]][1]
                    arc = arcmin
                    index = rr
            else:
                for i in range(0, len(free)):
                    aa = free[i]
                    # print(load[k])
                    if load[k] + map[aa[0]][aa[1]][1] <= CAPACITY:
                        if shortestPath[now][aa[0]] <= shortestPath[now][aa[1]]:
                            dmin = shortestPath[now][aa[0]]
                            arcmin = (aa[0], aa[1])
                        else:
                            dmin = shortestPath[now][aa[1]]
                            arcmin = (aa[1], aa[0])

                        if dmin < d:
                            d = dmin
                            q = map[aa[0]][aa[1]][1]
                            arc = arcmin
                            index = i
                        elif dmin == d:
                            if better(arcmin, arc, load[k], now):
                                arc = arcmin
                                q = map[aa[0]][aa[1]][1]
                                d = dmin
                                index = i

            if d != N:
                if shortestPath[now][arc[0]] == shortestPath[now][depot] + shortestPath[depot][arc[0]] and shortestPath[now][depot] != 0 and shortestPath[depot][arc[0]] != 0:
                    break
                else:
                    now = arc[1]
                    Route[k].append(arc)
                    if index != -1:
                        free.pop(index)
                    load[k] += q
                    cost[k] += d + map[arc[0]][arc[1]][0]
            else:
                break
        cost[k] += shortestPath[Route[k][len(Route[k])-1][1]][depot]
        total += cost[k]
    return Route, total


def pathScanningPro(start, SEED, TIME):
    random.seed(SEED)
    bestpop = []
    pop = []    # Set the current population pop = ∅
    ubtrial = 1000000  # trial's num
    l = N
    limit = TIME
    for i in range(ubtrial):
        Route, totalcost = pathscan(map)
        if not CloneSimple(pop, Route, totalcost):
            pop.append((Route, totalcost))
            if totalcost < l:
                bestpop = (Route, totalcost)
                l = totalcost
                print(i, l)
        # if time.time() - start > limit - 1:  # limit:
        #     printformat(bestpop)
        #     break


def calCost(sont):
    cost = 0
    now = DEPOT
    if type(sont) != list:
        # print(now,sont)
        cost += shortestPath[now][sont[0]] + map[sont[0]][sont[1]][0]
        now = sont[1]
    else:
        for i in range(len(sont)):
            # print(sont)
            cost += shortestPath[now][sont[i][0]] + \
                map[sont[i][0]][sont[i][1]][0]
            now = sont[i][1]
    cost += shortestPath[now][DEPOT]
    return cost


def calTT(solution):
    cost = 0
    if type(solution) != list:
        solution = [solution]
    for i in range(len(solution)):
        #print (i, solution[i],solution)
        cost += calCost(solution[i])
    return cost


def CloneSimple(pop, Route, totalcost):
    for i in range(len(pop)):
        if totalcost == pop[i][1]:
            return True
    return False


def printformat(bestpop):
    print('s ', end='')
    cnt = 0
    for r in bestpop[0]:
        print('0,', end='')
        for edge in r:
            print('(%d,%d)' % (edge[0], edge[1]), end=',')
        if cnt == len(bestpop[0]) - 1:
            print('0')
        else:
            print('0,', end='')
        cnt += 1
    print('q %d' % bestpop[1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str,
                        default='./CARP_samples/egl-e1-A.dat')
    parser.add_argument('-t', type=int, default=600)
    parser.add_argument('-s', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    filepath = args.filepath
    TIME = args.t
    SEED = args.s
    # start = 0
    # filepath = './CARP/CARP_samples/lab.dat'
    # SEED = 0
    # TIME = 500
    map = readFile(filepath)
    shortestPath = dijkstra(child)
    pathScanningPro(start, SEED, TIME)
