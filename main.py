import csv
import time
import random
import tracemalloc

import numpy as np
from memory_profiler import profile, memory_usage
global top_stats

# Load matrix of path costs from txt data file
def load_graph(filename):
    datafile = open(filename, "r")
    lines = datafile.read().splitlines()
    matrix = []
    for i in range(1, len(lines)):
        matrix.append(list(map(int, lines[i].split())))
    datafile.close()
    return matrix


def load_initialization_file(filename):
    init_file = open(filename, "r")
    configuration = init_file.read().splitlines()
    init_file.close()
    return configuration


# Function to calculate the total distance of the path
def calculate_distance(solution, graph):
    dist = 0
    for i in range(len(solution)):
        if i != len(solution) - 1:
            dist += graph[solution[i], solution[i + 1]]
        else:
            dist += graph[solution[i], solution[0]]
    return dist


# Function to generate a random solution
def random_solution(graph):
    solution = [i for i in range(len(graph))]
    random.shuffle(solution)
    return solution


# Function to generate a neighbour by switching the position of two randomly selected cities
def neighbour(solution):
    neighbour = solution[:]
    x, y = random.sample(range(len(neighbour)), 2)
    neighbour[x], neighbour[y] = neighbour[y], neighbour[x]
    return neighbour


# Function to run the simulation
#@profile
def simulated_annealing(temp, cooling_rate, iterations, graph):
    solution = random_solution(graph)
    best_solution = solution  # [:]
    for i in range(iterations):
        new_solution = neighbour(solution)
        random_solution_distance = calculate_distance(solution, graph)
        best_distance = calculate_distance(best_solution, graph)

        delta = calculate_distance(new_solution, graph) - random_solution_distance

        # If solution is worse or probability is very high
        if delta < 0 or random.random() < np.exp(-delta / temp):
            solution = new_solution  # [:]
        if random_solution_distance < best_distance:
            best_solution = solution  # [:]
        temp *= (1 - cooling_rate)
    return best_solution


def test_algorithm(configuration):
    with open(configuration[len(configuration) - 1], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time [ns]', 'correct result', 'memory used'])
    tracemalloc.start()
    for testing_case in range(len(configuration) - 1):
        with open(configuration[len(configuration) - 1], 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([configuration[testing_case]])
        testing_case_data = configuration[testing_case].split()
        file_name = testing_case_data[0]
        number_of_tests = int(testing_case_data[1])
        min_path_weight = int(testing_case_data[2])


        if file_name == "tsp_6_1.txt" or file_name == "tsp_6_2.txt":
            for test in range(number_of_tests):
                twenty_repetitions_time = 0
                for repetition in range(20):
                    start = time.time_ns()

                    graph = np.array(load_graph("TestData/" + file_name))

                    best_solution = simulated_annealing(35, 0.001, 5000, graph)
                    distance = calculate_distance(best_solution, graph)
                    print("Best solution: ", best_solution)
                    print("Distance: ", distance)

                    # optimized_graph, cost_of_graph = optimize_graph(load_graph("TestData/" + file_name))
                    # counted_min_path_weight, counted_path = find_path(optimized_graph, cost_of_graph)
                    # TODO

                    print("Analyzing " + file_name + "...")

                    if distance == min_path_weight:
                        print("Good job, that's right.")
                        correct = "tak"
                    else:
                        print("The algorithm is wrong. Check the minimal path weight.")
                        correct = "nie"

                    # if counted_path == min_path:
                    #     print("Good job, that's right.")
                    # else:
                    #     print("The algorithm is wrong. Check the minimal path")

                    end = time.time_ns()
                    twenty_repetitions_time += end - start
                with open(configuration[len(configuration) - 1], 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([str(twenty_repetitions_time), correct])
        else:
            for test in range(number_of_tests):
                print("kupa" + str(memory_usage()))
                operation_time = 0
                start = time.time_ns()

                graph = np.array(load_graph("TestData/" + file_name))

                # tracemalloc.start()
                best_solution = simulated_annealing(35, 0.001, 5000, graph)
                distance = calculate_distance(best_solution, graph)
                current, peak = tracemalloc.get_traced_memory()
                #print(f"Currentttttttttttt memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
                # print("AAAAAAAAAAAAAAAAAA")
                # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                # tracemalloc.stop()



                print("Best solution: ", best_solution)
                print("Distance: ", distance)

                print("Analyzing " + file_name + "...")

                if distance == min_path_weight:
                    print("Good job, that's right.")
                    correct = "tak"
                else:
                    print("The algorithm is wrong. Check the minimal path weight.")
                    correct = "nie"

                end = time.time_ns()
                operation_time += end - start
                with open(configuration[len(configuration) - 1], 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([str(operation_time), correct])
        snapshot = tracemalloc.take_snapshot()
        global top_stats
        top_stats = snapshot.statistics('lineno')


if __name__ == "__main__":
    configuration = load_initialization_file("init.ini")
    test_algorithm(configuration)
    print("dupa" + str(memory_usage()))
    print("[ Top 10 ]")
    global top_stats
    for stat in top_stats: # [:10]:
        print(stat.size)