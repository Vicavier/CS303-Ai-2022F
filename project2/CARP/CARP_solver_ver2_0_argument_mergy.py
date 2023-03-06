import numpy as np
import argparse
N = 2e9


def dijkstra(child: list):
    n = len(child)
    father = np.zeros((n, n), dtype=int)
    shortestPath = np.zeros((n, n), dtype=int)
    for i in range(n):
        visited = [0] * n
        dis = [N] * n
        f = [0] * n
        visited[i] = 1
        dis[i] = 0
        heap = []
        heap.append(i + 1)
        while len(heap) != 0:
            index = 0
            min = N
            for _ in range(len(heap)):
                if dis[heap[_] - 1] < min:
                    index = _
                    min = dis[heap[_] - 1]

            cur = heap[index]
            heap.pop(index)
            for j in child[cur - 1]:
                if visited[j[0] - 1] == 0:
                    visited[j[0] - 1] = 1
                    dis[j[0] - 1] = dis[cur - 1] + j[1]
                    heap.append(j[0])
                    f[j[0] - 1] = cur
                else:
                    if dis[j[0] - 1] > dis[cur - 1] + j[1]:
                        dis[j[0] - 1] = dis[cur - 1] + j[1]
                        f[j[0] - 1] = cur
        shortestPath[i] = dis
        father[i] = f
    return shortestPath, father


def initial(n, require, shortestPath, deport):
    route = []
    load = []
    cost = []
    for i in range(n):
        route.append(require[i][0])
        cost.append(shortestPath[deport - 1][require[i][0][0] - 1] +
                    require[i][1] + shortestPath[require[i][0][1] - 1][deport - 1])
        load.append(require[i][2])
    return route, cost, load


def argument(n, Q, initial_route, initial_cost, initial_load, father, deport):
    route = []
    cost = []
    load = []
    for k in range(len(initial_route)):
        Rk = [initial_route[k]]
        required = initial_route[k]
        if required != []:
            # TODO:assume that r=[(x,y)]: expand the route by inserting all arcs of SP_{1x} before (x,y)
            # and all acrs of SP_{y1} after (x,y)
            beg = required[0]
            end = required[1]
            temp = deport
            SP_y1 = []
            while father[deport - 1][beg - 1] != 0:
                Rk.insert(0, (father[deport - 1][beg - 1], beg))
                beg = father[deport - 1][beg - 1]
            while father[end - 1][temp - 1] != 0:
                SP_y1.append((father[end - 1][temp - 1], temp))
                temp = father[end - 1][temp - 1]
            SP_y1.reverse()
            Rk += SP_y1
            serviced = [0] * len(Rk)
            serviced[Rk.index(required)] = 1
            for p in range(k + 1, len(initial_route)):
                Rp = initial_route[p]
                if Rp != [] and initial_load[k] + initial_load[p] <= Q:
                    v = Rp[0]
                    w = Rp[1]
                    if (v, w) in Rk:
                        serviced[Rk.index((v, w))] = 1
                        initial_load[k] += initial_load[p]
                        initial_route[p] = []
                    if (w, v) in Rk:
                        serviced[Rk.index((w, v))] = 1
                        initial_load[k] += initial_load[p]
                        initial_route[p] = []
            cost.append(initial_cost[k])
            load.append(initial_load[k])
            tempRk = []
            for i in range(len(serviced)):
                if serviced[i] == 1:
                    tempRk.append(Rk[i])
            route.append(tempRk)

    return route, cost, load


def getA(argument_route, shortestPath):
    A = {}
    l = len(argument_route)
    for i in range(l - 1):
        for j in range(i + 1, l):
            A[(argument_route[i][-1][1], argument_route[j][0][0])
              ] = shortestPath[argument_route[i][-1][1] - 1][argument_route[j][0][0] - 1]
            A[(argument_route[i][0][0], argument_route[j][0][0])
              ] = shortestPath[argument_route[i][0][0] - 1][argument_route[j][0][0] - 1]
            A[(argument_route[i][0][0], argument_route[j][-1][1])
              ] = shortestPath[argument_route[i][0][0] - 1][argument_route[j][-1][1] - 1]
            A[(argument_route[i][-1][1], argument_route[j][-1][1])
              ] = shortestPath[argument_route[i][-1][1] - 1][argument_route[j][-1][1] - 1]
    return A


def merge(A_, Q, argument_route, argument_cost, argument_load, shortestPath):
    for arc in A_:
        for k in range(len(argument_route) - 1):
            Rk = argument_route[k]
            if Rk != []:
                for p in range(k + 1, len(argument_route)):
                    Rp = argument_route[p]
                    if Rp != []:
                        if arc[0] in (Rk[0][0], Rk[-1][1]) and arc[1] in (Rp[0][0], Rp[-1][1]):
                            if argument_load[k] + argument_load[p] <= Q:
                                if arc[0] == Rk[0][0]:
                                    Rk.reverse()
                                if arc[1] == Rp[-1][1]:
                                    Rp.reverse()
                                Rk += Rp
                                argument_load[k] += argument_load[p]
                                s_ij = (shortestPath[arc[0] - 1][deport - 1] +
                                        shortestPath[deport - 1][arc[1] - 1] - shortestPath[arc[0]-1][arc[1]-1])
                                argument_cost[k] += argument_cost[p] - s_ij
                                argument_route[p] = []
    route = []
    cost = []
    for i in range(len(argument_route)):
        if argument_route[i] != []:
            route.append(argument_route[i])
            cost.append(argument_cost[i])
    return route, cost


def argument_merge(deport, Q, require, shortestPath, father):
    # Initial routes
    n = int(len(require)/2)
    # initial phase
    initial_route, initial_cost, initial_load = initial(
        n, require, shortestPath, deport)

    # Argument phase
    argument_route, argument_cost, argument_load = argument(
        n, Q, initial_route, initial_cost, initial_load, father, deport)

    # Merge phase

    A = getA(argument_route, shortestPath)

    A_ = np.array(sorted(A.items(),  key=lambda v: v[1]), dtype=object)[:, 0]
    # A_ = np.array(sorted(A.items(),  key=lambda v: v[1]), dtype=object)
    # print(A_)
    merge_route, merge_cost = merge(
        A_, Q, argument_route, argument_cost, argument_load, shortestPath)

    return merge_route, merge_cost


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str,
                        default='./CARP_samples/egl-e1-A.dat')
    parser.add_argument('-t', type=int, default=600)
    parser.add_argument('-s', type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    specification = []
    cnt = Q = 0
    deport = 1
    child = []
    required = []
    cost = 0
    with open(args.filepath, encoding='utf-8') as f:
        while cnt < 8:
            specification.append(f.readline().split(' : ')[1].strip())
            cnt += 1

        for _ in range(int(specification[1])):
            child.append([])

        deport = int(specification[2])
        required_edge = int(specification[3])
        Q = int(specification[6])

        required = [0] * 2 * required_edge

        while cnt >= 8:
            line = f.readline()
            if line == 'END':
                break
            elif line == 'NODES       COST         DEMAND\n':
                cnt += 1
                continue
            else:
                data = line.split('   ')
                a = int(data[0])
                b = int(data[1])
                cost = int(data[2])
                demand = int(data[4].strip())
                child[a - 1].append((b, cost))
                child[b - 1].append((a, cost))
                if demand != 0:
                    required[cnt - 9] = [(a, b), cost, demand]
                    required[cnt - 9 + required_edge] = [(b, a), cost, demand]
                    cnt += 1
    shortestPath, father = dijkstra(child)
    route, cost = argument_merge(
        deport, Q, required, shortestPath, father)

    cost_sum = sum(cost)
    print('s ', end='')
    cnt = 0
    for r in route:
        print('0,', end='')
        for edge in r:
            print('(%d,%d)' % (edge[0], edge[1]), end=',')
        if cnt == len(route) - 1:
            print('0')
        else:
            print('0,', end='')
        cnt += 1
    print('q %d' % cost_sum)
