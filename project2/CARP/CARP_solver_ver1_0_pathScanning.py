import numpy as np
import argparse
N = 2e9


def dijkstra(child: list):
    n = len(child)
    shortestPath = np.zeros((n, n), dtype=int)
    for i in range(n):
        visited = [0] * n
        dis = [N] * n
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
                else:
                    if dis[j[0] - 1] > dis[cur - 1] + j[1]:
                        dis[j[0] - 1] = dis[cur - 1] + j[1]
        shortestPath[i] = dis
    return shortestPath


def pathScanning(depot, Q, required: list, shortestPath):
    free = required.copy()
    route = []
    total_cost = 0
    while len(free) != 0:
        load = cost = 0
        route_k = []
        end = depot
        while True:
            d = N
            u = None
            for _ in free:  # [[a,b],[cost,demand]]
                if _[1][1] + load <= Q:
                    if shortestPath[end-1][_[0][0]-1] < d:
                        d = shortestPath[end-1][_[0][0]-1]
                        u = _
            if d == N:
                cost += shortestPath[end - 1][deport - 1]
                route.append(route_k)
                total_cost += cost
                break
            route_k.append((u[0][0], u[0][1]))
            end = u[0][1]
            free.remove(u)
            u[0].reverse()
            free.remove(u)
            cost += d + u[1][0]
            load += u[1][1]

    return route, total_cost


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
        Q = int(specification[6])

        while cnt >= 8:
            line = f.readline()
            if line == 'END':
                break
            elif line == 'NODES       COST         DEMAND\n':
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
                    required.append([[a, b], [cost, demand]])
                    required.append([[b, a], [cost, demand]])
    shortestPath = dijkstra(child)
    route, cost = pathScanning(deport, Q, required, shortestPath)
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
    print('q %d' % cost)
