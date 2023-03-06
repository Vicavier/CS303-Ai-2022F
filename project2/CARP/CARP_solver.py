import random
import time
import argparse
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


def initialization(start):
    random.seed(SEED)
    pop = []    # Set the current population pop = ∅
    ubtrial = 1000000  # trial's num
    l = N
    limit = TIME*1/5
    for i in range(ubtrial):
        Route, totalcost = pathScanning(map)
        if not simple_exist(pop, totalcost):
            pop.append((Route, totalcost))
            if totalcost < l:
                l = totalcost
                # print(i, l)
        if time.time() - start > limit:  # limit:
            break
    pop.sort(key=lambda x: x[1], reverse=False)
    return pop


def pathScanning(map):
    depot = DEPOT
    free = copy.deepcopy(Required)  # Copy all required arcs in a list {free}
    k = -1
    Route = []
    load = []
    cost = []
    total = 0  # cost
    while len(free) != 0:
        # initialize the kth route
        k += 1
        Route.append([])
        load.append(0)
        cost.append(0)
        end = depot
        arc = (0, 0)
        while True:
            c = N   # cost
            index = -1
            d = 0   # demand
            r = random.random()
            if r < 0.5 and len(Route[k]) == 0:
                if len(free) == 0:
                    break
                rr = int(random.random()*(len(free)))   # 从free中任选一个
                cur = free[rr]
                if load[k] + map[cur[0]][cur[1]][1] <= CAPACITY:
                    if shortestPath[end][cur[0]] <= shortestPath[end][cur[1]]:
                        cmin = shortestPath[end][cur[0]]
                        arcmin = (cur[0], cur[1])
                    else:
                        cmin = shortestPath[end][cur[1]]
                        arcmin = (cur[1], cur[0])
                    c = cmin
                    d = map[cur[0]][cur[1]][1]
                    arc = arcmin
                    index = rr

            else:
                for i in range(0, len(free)):
                    cur = free[i]
                    if load[k] + map[cur[0]][cur[1]][1] <= CAPACITY:
                        if shortestPath[end][cur[0]] <= shortestPath[end][cur[1]]:
                            cmin = shortestPath[end][cur[0]]
                            arcmin = (cur[0], cur[1])
                        else:
                            cmin = shortestPath[end][cur[1]]
                            arcmin = (cur[1], cur[0])

                        if cmin < c:
                            c = cmin
                            d = map[cur[0]][cur[1]][1]
                            arc = arcmin
                            index = i
                        elif cmin == c:
                            # arc 为之前的解， arcmin是当前需better的，随机使用rule判断arc和arcmin谁更优
                            if better(arcmin, arc, load[k], end):
                                arc = arcmin
                                d = map[cur[0]][cur[1]][1]
                                c = cmin
                                index = i

            if c != N:
                # cut branches
                if shortestPath[end][arc[0]] == shortestPath[end][depot] + shortestPath[depot][arc[0]] and shortestPath[end][depot] != 0 and shortestPath[depot][arc[0]] != 0:
                    break
                else:
                    end = arc[1]
                    Route[k].append(arc)
                    if index != -1:
                        free.pop(index)
                    load[k] += d
                    cost[k] += c + map[arc[0]][arc[1]][0]
            else:
                break
        cost[k] += shortestPath[Route[k][len(Route[k])-1][1]][depot]
        total += cost[k]
    return Route, total


def better(arcmin, arc, load, now):
    rule = random.random()
    if rule < 0.15:
        if shortestPath[arcmin[1]][DEPOT] > shortestPath[arc[1]][DEPOT]:
            return True
        else:
            return False
    elif rule < 0.3:
        if shortestPath[arcmin[1]][DEPOT] < shortestPath[arc[1]][DEPOT]:
            return True
        else:
            return False
    elif rule < 0.6:  # demand/cost
        if map[arcmin[0]][arcmin[1]][1]/(shortestPath[now][arcmin[0]]+map[arcmin[0]][arcmin[1]][0]) > map[arc[0]][arc[1]][1]/(shortestPath[now][arc[0]]+map[arc[0]][arc[1]][0]):
            return True
        else:
            return False
    elif rule < 0.7:
        if map[arcmin[0]][arcmin[1]][1]/(shortestPath[now][arcmin[0]]+map[arcmin[0]][arcmin[1]][0]) < map[arc[0]][arc[1]][1]/(shortestPath[now][arc[0]]+map[arc[0]][arc[1]][0]):
            return True
        else:
            return False
    elif rule < 1:
        if load < CAPACITY / 2 and shortestPath[arcmin[1]][DEPOT] > shortestPath[arc[1]][DEPOT]:
            return True
        elif load > CAPACITY / 2 and shortestPath[arcmin[1]][DEPOT] < shortestPath[arc[1]][DEPOT]:
            return True
        else:
            return False


def simple_exist(pop, totalcost):
    for i in range(len(pop)):
        if totalcost == pop[i][1]:
            return True
    return False

# --------------------MEANS Module------------------


def MEANS(pop, start):
    bestpop = []
    GM = 500  # number of population
    psizem = 30
    gm_counter = 0
    sample = 0
    while gm_counter < GM:
        # print("第 %d 代"%gm_counter)
        if gm_counter == 0:
            startt = time.time()

        popt = copy.deepcopy(pop)
        psize = len(pop)
        opsize = 6 * len(pop)
        i = 0
        j = 0
        while i < opsize and j < 500:
            # Randomly select two different solutions S1 and S2 as the parents from popt
            a = int(random.random()*(psize))
            b = int(random.random()*(psize))
            while b == a:
                b = int(random.random() * (psize))
            S1 = pop[a]
            S2 = pop[b]
            # Apply the crossover operator to S1 and S2 to generate son[i]
            son = SBX(S1, S2)  # two existed solutions

            if not is_in_popt(popt, son):
                popt.append(son)
                i += 1
            j += 1

        popt.sort(key=lambda x: x[1], reverse=False)
        if len(popt) > psizem:
            pop = popt[0:psizem+1]
        else:
            pop = popt

        bestpop = pop[0]
        # printformat(bestpop[0], bestpop[1])
        # print('time: %.2f' % (time.time() - start))

        if gm_counter == 0:
            sample = time.time() - startt
        if time.time() - start > TIME-sample-1:
            break
        gm_counter += 1

    # print("end")
    printformat(bestpop[0], bestpop[1])


def SBX(S1, S2):
    S1_t = copy.deepcopy(S1)
    S2_t = copy.deepcopy(S2)
    # randomly selects two routes R1 and R2 from them, respectively
    index1 = int(random.random() * (len(S1_t[0])))
    index2 = int(random.random() * (len(S2_t[0])))
    # random route in solution
    R1 = copy.deepcopy(S1_t[0][index1])
    R2 = copy.deepcopy(S2_t[0][index2])

    if len(R1) <= 1:
        return S2_t
    if len(R2) <= 1:
        return S1_t
    x = 0
    y = 0
    while x == 0 or x == len(R1):
        x = int(random.random() * (len(R1)))
    while y == 0 or y == len(R2):
        y = int(random.random() * (len(R2)))

    # both R1 and R2 are further randomly split into two subroutes, say R1 = (R11, R12) and R2 = (R21, R22)
    R11 = R1[0:x]
    R12 = R1[x:]
    R21 = R2[0:y]
    R22 = R2[y:]

    # a new route is obtained by replacing R12 with R22
    new_route1 = R11 + R22
    new_route2 = R21 + R12
    # remove duplicated edges and insert lacked edges
    son1 = motify_new_route(copy.deepcopy(S1_t[0]), index1, new_route1)
    son2 = motify_new_route(copy.deepcopy(S2_t[0]), index2, new_route2)
    # remain the better son
    cost1 = cal_totalCost(son1)
    cost2 = cal_totalCost(son2)
    if cost1 < cost2:
        newson = (son1, cost1)
        return newson
    else:
        newson = (son2, cost2)
        return newson


def motify_new_route(S, route_index, new_routet):
    S_r = copy.deepcopy(S)  # just a copy in case of return
    S_t = copy.deepcopy(S)
    old_route = copy.deepcopy(S[route_index])  # 最初始要更改的solution里的route
    new_route = copy.deepcopy(new_routet)

    # step 1: remove duplicate edge
    duplicate = []
    # step 1.1: find duplicated edges
    for i in range(len(new_route)):
        if (new_route[i] not in old_route) and ((new_route[i][1], new_route[i][0]) not in old_route):
            duplicate.append(new_route[i])
    # step 1.2: remove the edges that appear more than once in the new_route
    index = 0
    while index < len(new_route)-1:
        for j in range(index+1, len(new_route)):
            if (new_route[index] == new_route[j] or (new_route[index][1], new_route[index][0]) == new_route[j]):
                new_route.pop(j)
                break
        index += 1

    S_t[route_index] = new_route.copy()

    # step 1.3: remove edges that in duplicate[] in orther route
    for i in range(len(duplicate)):
        for j in range(len(S_t)):
            if j == route_index:
                continue
            else:
                k = 0
                while k < len(S_t[j]):
                    if (S_t[j][k] == duplicate[i] or (duplicate[i][1], duplicate[i][0]) == S_t[j][k]):
                        S_t[j].pop(k)
                    else:
                        k += 1

    # step 2: add lack edges
    # step 2.1: find lack edges
    lack = findlack(old_route, new_route)
    # step 2.2: check whether S1_t[index_r1]'s demand is unsatisfied
    while cal_demand(S_t[route_index]) > CAPACITY:
        popindex = int(random.random()*len(S_t[route_index]))
        node = S_t[route_index].pop(popindex)
        lack.append(node)

    # step 2.3: add
    '''
    each missing task is re-inserted into such a position
that re-insertion into any other position will not induce both
lower additional cost and smaller violation of the capacity
constraints. If multiple positions satisfy this condition, one of
them will be chosen arbitrarily.
'''
    # success = 1  # 表示插入成功
    # demand_remain = []
    # for i in range(len(S1_t)):
    #     demand_remain.append(CAPACITY - cal_demand(S1_t[i]))

    # for l in lack:
    #     optional = [i for i, x in enumerate(
    #         demand_remain) if x > map[l[0]][l[1]][1]]   # 当前任务能插入的位置
    #     if len(optional) == 0:  # 说明已经没有路径能插入当前任务，但是lack没有为空
    #         success = 0  # 插入失败
    #         break
    #     else:
    #         # TODO:插入使得增加的cost最小，并且最大的demand容差
    #         # insert_index = int(random.random()*len(optional))  # 随机选择一条插入
    #         insert_index = 0
    #         cost = N
    #         for position in optional:
    #             S1_t[position].append(l)
    #             min_cost = cal_totalCost(S1_t)
    #             if min_cost <= cost:
    #                 cost = min_cost
    #                 if demand_remain[position] > demand_remain[insert_index]:
    #                     insert_index = position
    #             else:
    #                 S1_t[position].pop()
    uppp = 0
    success = 0
    while len(lack) != 0 and uppp < 6 and success == 0:
        i = 0
        success = 1
        trailt = 0
        upper = max(len(lack)*len(lack)/2, 10)
        temp_route = copy.deepcopy(S_t)
        lackt = lack.copy()

        while lackt != [] and trailt < upper:
            isinsert = 0
            trailt = 0
            obj = lackt[0]
            while isinsert == 0 and trailt < len(temp_route):
                # randomly choose a route to insert
                k = int(random.random()*len(temp_route))
                # check demand
                if (map[obj[0]][obj[1]][1] + cal_demand(temp_route[k])) <= CAPACITY:
                    cost = N
                    indexn = 0
                    arc = lackt[i]
                    reverse = 0
                    # if demand is satisfied, insert (v,w) and (w,v) in each position in the route, and choose the least cost
                    for j in range(0, len(temp_route[k]) + 1):
                        sont = copy.deepcopy(temp_route[k])
                        sont.insert(j, arc)
                        costt = cal_cost(sont)
                        if costt < cost:
                            cost = costt
                            indexn = j
                    arc = (arc[1], arc[0])
                    for j in range(0, len(temp_route[k]) + 1):
                        sont = copy.deepcopy(temp_route[k])
                        sont.insert(j, arc)
                        costt = cal_cost(sont)
                        if costt < cost:
                            cost = costt
                            indexn = j
                            reverse = 1
                    if reverse == 1:
                        temp_route[k].insert(indexn, arc)
                    else:
                        temp_route[k].insert(indexn, lackt[i])
                    lackt.pop(0)
                    isinsert += 1
                    break
                trailt += 1
            if lackt == []:
                break
            if isinsert == 0:
                success = 0
                break
            trailt += 1
        if lackt == []:
            lack = lackt.copy()
            S_t = copy.deepcopy(temp_route)
        else:
            uppp += 1
            continue

    if lack == []:
        return S_t
    else:
        return S_r


def cal_demand(son):
    if type(son) != list:
        son = [son]
    cost = 0
    for i in range(len(son)):
        cost += map[son[i][0]][son[i][1]][1]
    return cost


def findlack(old_route, new_route):
    lack = []
    # edges that old_route has but new_route hasn't
    for e in old_route:
        if (e not in new_route) and ((e[1], e[0]) not in new_route):
            lack.append(e)
    # edges that violate the CAPACITY
    while cal_demand(new_route) > CAPACITY:
        random_pop_index = int(random.random()*len(new_route))
        lack.append(new_route.pop(random_pop_index))
    return lack


def cal_cost(route):
    cost = 0
    now = DEPOT

    for i in range(len(route)):
        cost += shortestPath[now][route[i][0]] + \
            map[route[i][0]][route[i][1]][0]
        now = route[i][1]
    cost += shortestPath[now][DEPOT]
    return cost


def cal_totalCost(solution):
    cost = 0

    for i in range(len(solution)):
        #print (i, solution[i],solution)
        cost += cal_cost(solution[i])
    return cost


def is_in_popt(pop, son):
    for i in range(len(pop)):
        flag_cost = 0
        if son[1] == pop[i][1]:
            flag_cost = 0
            for j in range(len(son[0])):
                flag = 0
                for k in range(len(pop[i][0])):
                    if son[0][j] == pop[i][0][k]:
                        flag = 1
                if flag == 0:
                    break
            if flag == 1:
                flag_cost = 1
        if flag_cost == 1:
            # print('厚礼谢!')
            return True
    return False


def printformat(route, cost):
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
    map = readFile(filepath)
    shortestPath = dijkstra(child)
    # print(shortestPath)
    bestpop = []
    initial_pop = initialization(start)
    MEANS(initial_pop, start)
