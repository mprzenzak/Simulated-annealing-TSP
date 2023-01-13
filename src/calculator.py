import math
import os
import psutil
import random
import time


def calculate_temperature(type, initial_temperature, era, alpha):
    if type == 'Geometrical':
        return initial_temperature * alpha ** era
    elif type == 'Boltzmann':
        return initial_temperature / (1 + math.log(era))


def calc_cost(path, matrix):
    # greedy
    cost = matrix[0][path[0]]
    for i in range(len(path) - 1):
        cost += matrix[path[i]][path[i + 1]]

    cost += matrix[path[-1]][0]
    return cost


# TODO steepest

def run_test(matrix, config):
    temperature, minimal_temperature, cooling_rate, max_eras, era_length, solution_in_neighbourhood_type = config[
                                                                                                               'Temperature'], \
                                                                                                           config[
                                                                                                               'Minimal_temperature'], \
                                                                                                           config[
                                                                                                               'Cooling_Rate'], \
                                                                                                           config[
                                                                                                               'Eras'], \
                                                                                                           config[
                                                                                                               'Era_length'], \
                                                                                                           config[
                                                                                                               'Solution_in_neighbourhood']

    start_time = time.time_ns()

    current_path = get_random_path(matrix)
    current_cost = calc_cost(current_path, matrix)

    best_path = current_path
    best_cost = current_cost

    era = 1
    while temperature > minimal_temperature and era < max_eras + 1:
        for i in range(era_length):
            new_path = generate_neighbour(current_path, solution_in_neighbourhood_type)
            new_cost = calc_cost(new_path, matrix)

            if new_cost < current_cost:
                current_path = new_path
                current_cost = new_cost
                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost

            else:
                acceptance_probability = math.exp(-(new_cost - best_cost) / temperature)
                if random.random() < acceptance_probability:
                    current_path = new_path
                    current_cost = new_cost

        temperature = calculate_temperature(config['Cooling_Type'], config['Temperature'], era, cooling_rate)

        new_cost /= era_length
        era += 1
    stop_time = time.time_ns()

    return {
        'time': stop_time - start_time,
        'memory_usage': psutil.Process(os.getpid()).memory_info().rss / 1000000,
        'solution': best_cost,
        'path': (0, *best_path, 0)
    }


def get_random_path(matrix):
    path = []
    tmp_path = [v for v in range(1, len(matrix))]

    for _ in range(len(tmp_path)):
        next_point = tmp_path[random.randint(0, len(tmp_path) - 1)]
        path.append(next_point)
        tmp_path.remove(next_point)

    return tuple(path)


def generate_random_value(path, _not=None):
    val = random.randint(0, len(path) - 1)
    while (val == _not):
        val = random.randint(0, len(path) - 1)

    return val


def generate_neighbour(path, type):
    if type == "2swaps":
        path = list(path)
        index_1 = generate_random_value(path)
        index_2 = generate_random_value(path, _not=index_1)

        path[index_1], path[index_2] = path[index_2], path[index_1]

        return tuple(path)

    elif type == "arc":
        path = list(path)
        index_1 = generate_random_value(path)
        index_2 = generate_random_value(path, _not=index_1)

        new_path = []

        element1 = path[index_1]
        element2 = path[index_2]
        if index_1 < index_2:
            temp_path = path[index_1 + 1:index_2]
            temp_path.reverse()
            new_path += path[0:index_1]
            new_path.append(element2)
            new_path += temp_path
            new_path.append(element1)
            new_path += path[index_2 + 1:]
        else:
            temp_path = path[index_2 + 1:index_1]
            temp_path.reverse()
            new_path += path[0:index_2]
            new_path.append(element2)
            new_path += temp_path
            new_path.append(element1)
            new_path += path[index_1 + 1:]
        return tuple(new_path)
