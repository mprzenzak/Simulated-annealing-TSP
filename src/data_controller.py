import os

import networkx
import tsplib95


def load_config():
    config = {}
    lines = get_file_lines('config.ini')
    for line in lines:
        if line.startswith('#') or len(line.strip()) == 0:
            continue

        key_value_split = line.split('=')
        comment_split = key_value_split[1].split('#')
        config[key_value_split[0]] = parse(comment_split[0].strip())

    return config


def parse(value):
    if value in ['True', 'False']:
        return value == 'True'

    try:
        num = int(value)
        return num
    except:
        pass

    try:
        num = float(value)
        return num
    except:
        pass

    if value == '':
        return None

    return value


def str_to_tuple(string):
    tuple_ex = Exception(f'Podana wartosc "{string}" nie jest poprawnym tuplem')
    try:
        s = eval(string)
        if type(s) == tuple:
            return s
        raise tuple_ex
    except:
        raise tuple_ex


def read_matrix(filepath: str):
    if filepath.split('.')[-1] == 'txt':
        lines = get_file_lines(filepath)
        lines = [line for line in lines if len(line.strip().split(' ')) > 1]

        matrix = [[] for _ in range(len(lines))]

        for i, line in enumerate(lines):
            values = " ".join(line.split()).split(' ')

            if len(values) == 1 or len(values) == 0:
                continue

            for v in values:
                matrix[i].append(int(v))

        return matrix
    else:
        problem = tsplib95.load(filepath)
        graph = problem.get_graph()
        distance_matrix = networkx.to_numpy_array(graph)
        matrix = distance_matrix.tolist()
        return matrix


def print_matrix(matrix: list):
    for row in matrix:
        for val in row:
            print(val, end=' ')
        print()


def get_file_lines(filepath: str):
    if not os.path.exists(filepath):
        raise Exception(f"File doesn't exist: '{filepath}'")

    with open(filepath, 'r') as f:
        return f.read().splitlines()
