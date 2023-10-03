import numpy as np
import numba
import multiprocessing
import random
import time
from functools import wraps
import click

@click.group()
def cli():
    pass

def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print(f"fun: {f.__name__}, took: {te-ts} sec")
        return result
    return wrap

#@timing
def create_full_planar_graph(n:int):
    """
    :param n:int
    :return: np.array((n*n, n*n))
    """
    graph = np.zeros((n*n, n*n))

    for i in range(n*n):
        for j in range(n*n):
            if i != j:
                graph[i][j] = np.inf

    for i in range(n * n):
        for j in range(i, n * n - 1):
            if i == j and (j + 1) % n != 0:
                graph[i][j + 1] = random.randint(5, 25)

    for i in range(n * n):
        for j in range(i, n * n - n):
            if i == j:
                graph[i][j + n] = random.randint(5, 25)

    return graph

#@timing
@numba.njit(cache=True)
def create_full_planar_graph_jit(n:int):
    """
    :param n:int
    :return: np.array((n*n, n*n))
    """
    graph = np.zeros((n*n, n*n))

    for i in range(n*n):
        for j in range(n*n):
            if i != j:
                graph[i][j] = np.inf

    for i in range(n * n):
        for j in range(i, n * n - 1):
            if i == j and (j + 1) % n != 0:
                graph[i][j + 1] = random.randint(5, 25)

    for i in range(n * n):
        for j in range(i, n * n - n):
            if i == j:
                graph[i][j + n] = random.randint(5, 25)

    return graph

#@timing
@numba.njit(parallel = True, cache=True)
def create_full_planar_graph_parallel(n:int):
    """
    :param n:int
    :return: np.array((n*n, n*n))
    """
    graph = np.zeros((n*n, n*n))

    for i in numba.prange(n*n):
        for j in numba.prange(n*n):
            if i != j:
                graph[i][j] = np.inf

    for i in numba.prange(n * n):
        for j in numba.prange(i, n * n - 1):
            if i == j and (j + 1) % n != 0:
                graph[i][j + 1] = random.randint(5, 25)

    for i in numba.prange(n * n):
        for j in numba.prange(i, n * n - n):
            if i == j:
                graph[i][j + n] = random.randint(5, 25)

    return graph

#@timing
def Dijkstra_alg(graph, pos:int) -> (np.int64, np.array(int)):
    """
    Algorithm for finding the shortest paths between nodes in a weighted graph, which may represent, for example, road networks.
    :param graph: np.array((n,n))
    :param path: np.array(n)
    :param pos: int
    :return: len(path)
    """
    n = len(graph[0])
    distance = np.zeros(n)
    visited = np.zeros(n)
    path = np.zeros(n)

    for i in range(n):
        distance[i] = np.inf

    distance[pos] = 0
    for i in range(n - 1):
        min = np.inf
        # Из ещё не посещенных вершин находим вершину, имеющую минммальную метку
        for j in range(n):
            if not visited[i] and distance[i] <= min:
                min = distance[i]
                index = i  # Сохраняем индекс вершины
        visited[index] = True  # Помечаем, как посещенную
        for k in range(n):
            # Для каждого соседа найденной вершины, кроме отмеченных как посещённые,
            # рассмотрим новую длину пути, равную сумме значений текущей метки и длины ребра, соединяющего её с соседом.
            if not visited[k] and graph[index][k] != 0 and distance[index] != np.inf and (
                    distance[index] + graph[index][k] < distance[k]):
                distance[k] = distance[index] + graph[index][k]  # Заменяем значение метки

    if distance[n-1] == np.inf:
        return
    end = n-1
    begin = pos
    k = 1
    path[0] = end
    weight = distance[end]
    m = end

    while (end != begin):  # Пока не дошли до начальной вершины
        for i in range(n):
            if graph[i][end] != np.inf and graph[i][end] != 0:  # Если связь есть
                temp = weight - graph[i][end]  # Определяем вес пути из предыдущей вершины
                if temp == distance[i]:  # Если вес совпал с расчитанным, значит из этой вершины и был переход
                    weight = temp  # Сохраняем новый вес
                    end = i  # Сохраняем прндыдущую вершины
                    path[k] = i  # Записываем ее в массив
                    k += 1

    # print(f'Минимальной путь из {begin} в {m} с помощью алгоритма Дийкстры')
    # for i in range(k - 1, -1, -1):
    #     print(int(path[i]), end=" -> ")
    # print(m)
    return (k, path)

#@timing
@numba.njit(cache=True)
def Dijkstra_alg_jit(graph, pos:int) -> (np.int64, np.array(int)):
    """
    Algorithm for finding the shortest paths between nodes in a weighted graph, which may represent, for example, road networks.
    :param graph: np.array((n,n))
    :param path: np.array(n)
    :param pos: int
    :return: len(path)
    """
    n = len(graph[0])
    distance = np.zeros(n)
    visited = np.zeros(n)
    path = np.zeros(n, dtype=np.int64)

    for i in range(n):
        distance[i] = np.inf

    distance[pos] = 0
    for i in range(n - 1):
        min = np.inf
        # Из ещё не посещенных вершин находим вершину, имеющую минммальную метку
        for j in range(n):
            if not visited[i] and distance[i] <= min:
                min = distance[i]
                index = i  # Сохраняем индекс вершины
        visited[index] = True  # Помечаем, как посещенную
        for k in range(n):
            # Для каждого соседа найденной вершины, кроме отмеченных как посещённые,
            # рассмотрим новую длину пути, равную сумме значений текущей метки и длины ребра, соединяющего её с соседом.
            if not visited[k] and graph[index][k] != 0 and distance[index] != np.inf and (
                    distance[index] + graph[index][k] < distance[k]):
                distance[k] = distance[index] + graph[index][k]  # Заменяем значение метки

    if distance[n-1] == np.inf:
        return
    end = n-1
    begin = pos
    k = 1
    path[0] = end
    weight = distance[end]
    m = end

    while (end != begin):  # Пока не дошли до начальной вершины
        for i in range(n):
            if graph[i][end] != np.inf and graph[i][end] != 0:  # Если связь есть
                temp = weight - graph[i][end]  # Определяем вес пути из предыдущей вершины
                if temp == distance[i]:  # Если вес совпал с расчитанным, значит из этой вершины и был переход
                    weight = temp  # Сохраняем новый вес
                    end = i  # Сохраняем прндыдущую вершины
                    path[k] = i  # Записываем ее в массив
                    k += 1

    # print(f'Минимальной путь из {begin} в {m} с помощью алгоритма Дийкстры')
    # for i in range(k - 1, -1, -1):
    #     print(int(path[i]), end=" -> ")
    # print(m)
    return (k, path)

#@timing
@numba.njit(parallel=True, cache=True)
def Dijkstra_alg_parallel(graph, pos:int):
    """
    Algorithm for finding the shortest paths between nodes in a weighted graph, which may represent, for example, road networks.
    :param graph: np.array((n,n))
    :param path: np.array(n)
    :param pos: int
    :return: len(path)
    """
    n = len(graph[0])
    distance = np.zeros(n)
    visited = np.zeros(n)
    path = np.zeros(n, dtype=np.int64)

    for i in numba.prange(n):
        distance[i] = np.inf

    distance[pos] = 0
    for i in range(n - 1):
        min = np.inf
        # Из ещё не посещенных вершин находим вершину, имеющую минммальную метку
        for j in range(n):
            if not visited[i] and distance[i] <= min:
                min = distance[i]
                index = i  # Сохраняем индекс вершины
        visited[index] = True  # Помечаем, как посещенную
        for k in numba.prange(n):
            # Для каждого соседа найденной вершины, кроме отмеченных как посещённые,
            # рассмотрим новую длину пути, равную сумме значений текущей метки и длины ребра, соединяющего её с соседом.
            if not visited[k] and graph[index][k] != 0 and distance[index] != np.inf and (
                    distance[index] + graph[index][k] < distance[k]):
                distance[k] = distance[index] + graph[index][k]  # Заменяем значение метки

    if distance[n-1] == np.inf:
        return (np.int64(0), np.zeros(n, dtype=np.int64))
    end = n-1
    begin = pos
    k = 1
    path[0] = end
    weight = distance[end]
    m = end

    while (end != begin):  # Пока не дошли до начальной вершины
        for i in range(n):
            if graph[i][end] != np.inf and graph[i][end] != 0:  # Если связь есть
                temp = weight - graph[i][end]  # Определяем вес пути из предыдущей вершины
                if temp == distance[i]:  # Если вес совпал с расчитанным, значит из этой вершины и был переход
                    weight = temp  # Сохраняем новый вес
                    end = i  # Сохраняем прндыдущую вершины
                    path[k] = i  # Записываем ее в массив
                    k += 1

    # print(f'Минимальной путь из {begin} в {m} с помощью алгоритма Дийкстры')
    # for i in range(k - 1, -1, -1):
    #     print(int(path[i]), end=" -> ")
    # print(m)

    return (k, path)

def create_stream_matr(graph):
    n = len(graph[0])
    stream = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if graph[i][j] == np.inf:
                stream[i][j] = np.inf

    return stream

@numba.njit(cache=True)
def create_stream_matr_jit(graph):
    n = len(graph[0])
    stream = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if graph[i][j] == np.inf:
                stream[i][j] = np.inf

    return stream

@numba.njit(parallel=True, cache=True)
def create_stream_matr_parallel(graph):
    n = len(graph[0])
    stream = np.zeros((n,n))
    for i in numba.prange(n):
        for j in numba.prange(n):
            if graph[i][j] == np.inf:
                stream[i][j] = np.inf

    return stream

@timing
def max_stream_alg(graph) -> np.array((int, int)):
    """
    Algorithm for finding maximum stream in traphic
    :param graph: np.array((n,n))
    :return: stream: np.array((n,n))
    """
    stream = create_stream_matr(graph)
    while True:
        #Вызываем алгоритм Дейкстры для нахождения пути из истока в сток
        tuple = Dijkstra_alg(graph, 0)
        # Если такого пути нет, процесс завершаем
        if tuple == None:
            break
        path_len = tuple[0]
        path = tuple[1]

        #Находим величину del, которая насыщает одну из дуг найденного пути
        delete = np.inf
        mini = np.zeros(path_len-1)

        for i in range(path_len-1, 0, -1):
            k = int(path[i])
            j = int(path[i-1])
            mini[i-1] = graph[k][j] - stream[k][j]

        delete = min(mini)
        #print(f"del = {delete}")
        for i in range(path_len-1, 0, -1):
            k = int(path[i])
            j = int(path[i - 1])
            #Увеличиваем поток на найденную ранее величину
            stream[k][j] += delete
            #Если находим насыщенную дугу в потоке, то удаляем эту дугу из графа
            if graph[k][j] == stream[k][j]:
                graph[k][j] = np.inf

    return stream

@timing
@numba.njit(cache=True)
def max_stream_alg_jit(graph) -> np.array((int, int)):
    """
    Algorithm for finding maximum stream in traphic
    :param graph: np.array((n,n))
    :return: stream: np.array((n,n))
    """
    stream = create_stream_matr_jit(graph)
    while True:
        # Вызываем алгоритм Дейкстры для нахождения пути из истока в сток
        # Если такого пути нет, процесс завершаем
        try:
            tuple = Dijkstra_alg_jit(graph, 0)
            path_len, path = tuple
        except Exception:
            break
        #Находим величину del, которая насыщает одну из дуг найденного пути
        delete = np.inf
        mini = np.zeros(path_len-1)

        for i in range(path_len-1, 0, -1):
            k = path[i]
            j = path[i-1]
            mini[i-1] = graph[k][j] - stream[k][j]

        delete = min(mini)
        #print(f"del = {delete}")
        for i in range(path_len-1, 0, -1):
            k = path[i]
            j = path[i - 1]
            #Увеличиваем поток на найденную ранее величину
            stream[k][j] += delete
            #Если находим насыщенную дугу в потоке, то удаляем эту дугу из графа
            if graph[k][j] == stream[k][j]:
                graph[k][j] = np.inf

    return stream

@timing
@numba.njit(parallel=True, cache=True)
def max_stream_alg_parallel(graph) -> np.array((int, int)):
    """
    Algorithm for finding maximum stream in traphic
    :param graph: np.array((n,n))
    :return: stream: np.array((n,n))
    """
    stream = create_stream_matr_parallel(graph)
    while True:
        #Вызываем алгоритм Дейкстры для нахождения пути из истока в сток
        tuple = Dijkstra_alg_parallel(graph, 0)
        path_len, path = tuple

        # Если такого пути нет, процесс завершаем
        if path_len == 0:
            break

        #Находим величину del, которая насыщает одну из дуг найденного пути
        delete = np.inf
        mini = np.zeros(path_len-1)

        for i in range(path_len-1, 0, -1):
            k = path[i]
            j = path[i-1]
            mini[i-1] = graph[k][j] - stream[k][j]

        delete = min(mini)
        #print(f"del = {delete}")
        for i in range(path_len-1, 0, -1):
            k = path[i]
            j = path[i - 1]
            #Увеличиваем поток на найденную ранее величину
            stream[k][j] += delete
            #Если находим насыщенную дугу в потоке, то удаляем эту дугу из графа
            if graph[k][j] == stream[k][j]:
                graph[k][j] = np.inf

    return stream

@cli.command()
@click.option('--jit/--no-jit', default=False)
def jit_test(jit):
    if jit:
        click.echo(click.style('Running with JIT', fg='green'))
        graph = create_full_planar_graph_jit(100)
        max_stream_alg_jit(graph)
    else:
        click.echo(click.style('Running NO JIT', fg='red'))
        graph = create_full_planar_graph(100)
        max_stream_alg(graph)

@cli.command()
@click.option('--threads/--no-jit', default=False)
def threads_test(threads):
    if threads:
        click.echo(click.style('Running with multicore threads', fg='yellow'))
        graph = create_full_planar_graph_parallel(100)
        max_stream_alg_parallel(graph)
    else:
        click.echo(click.style('Running NO JIT', fg='red'))
        graph = create_full_planar_graph(100)
        max_stream_alg(graph)

if __name__ == '__main__':
    cli()

