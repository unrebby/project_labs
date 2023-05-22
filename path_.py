def path_Nikolskaya(path, maze_Nikolskaya):
    path_res = []
    for k in range(len(path) - 1):
        nodes = maze_Nikolskaya.keys()
        unvisited = {node: None for node in nodes}
        visited = {}
        current = path[k]
        currentDistance = 0
        unvisited[current] = currentDistance
        distance = 1

        all_path = list(list())

        while True:
            for neighbour in maze_Nikolskaya[current]:
                if neighbour not in unvisited: continue
                newDistance = currentDistance + distance
                if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                    unvisited[neighbour] = newDistance
                    all_path.append([current, neighbour])
            visited[current] = currentDistance
            del unvisited[current]
            if not unvisited: break
            candidates = [node for node in unvisited.items() if node[1]]
            current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]

        ans_path = ""
        now = path[k + 1]
        all_path_rev = list(reversed(all_path))
        for i in range(len(all_path_rev)):
            if all_path_rev[i][1] == now:
                ans_path += all_path_rev[i][1]
                now = all_path_rev[i][0]

        ans_path += path[k]
        path_res.append(ans_path[::-1])
    return path_res


def path_Berezhnoy(path, maze_Berezhnoy):
    path_res = []
    for k in range(len(path) - 1):
        nodes = maze_Berezhnoy.keys()
        unvisited = {node: None for node in nodes}
        visited = {}
        current = path[k]
        currentDistance = 0
        unvisited[current] = currentDistance
        distance = 1

        all_path = list(list())

        while True:
            for neighbour in maze_Berezhnoy[current]:
                if neighbour not in unvisited: continue
                newDistance = currentDistance + distance
                if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                    unvisited[neighbour] = newDistance
                    all_path.append([current, neighbour])
            visited[current] = currentDistance
            del unvisited[current]
            if not unvisited: break
            candidates = [node for node in unvisited.items() if node[1]]
            current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]

        ans_path = ""
        now = path[k + 1]
        all_path_rev = list(reversed(all_path))
        for i in range(len(all_path_rev)):
            if all_path_rev[i][1] == now:
                ans_path += all_path_rev[i][1]
                now = all_path_rev[i][0]

        ans_path += path[k]
        path_res.append(ans_path[::-1])
    return path_res


def path_Chelnok(path, maze_Chelnok):
    path_res = []
    for k in range(len(path) - 1):
        nodes = maze_Chelnok.keys()
        unvisited = {node: None for node in nodes}
        visited = {}
        current = path[k]
        currentDistance = 0
        unvisited[current] = currentDistance
        distance = 1

        all_path = list(list())

        while True:
            for neighbour in maze_Chelnok[current]:
                if neighbour not in unvisited: continue
                newDistance = currentDistance + distance
                if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                    unvisited[neighbour] = newDistance
                    all_path.append([current, neighbour])
            visited[current] = currentDistance
            del unvisited[current]
            if not unvisited: break
            candidates = [node for node in unvisited.items() if node[1]]
            current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]

        ans_path = ""
        now = path[k + 1]
        all_path_rev = list(reversed(all_path))
        for i in range(len(all_path_rev)):
            if all_path_rev[i][1] == now:
                ans_path += all_path_rev[i][1]
                now = all_path_rev[i][0]

        ans_path += path[k]
        path_res.append(ans_path[::-1])
    return path_res