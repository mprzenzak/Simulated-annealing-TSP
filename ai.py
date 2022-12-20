# simulated
# annealing
# algorithm
# solving
# traveling
# salesman
# problem in python.Assume
# there
# are
# 6
# nodes.
import math
# import packages
# import numpy as np
# import random
#
# # define the cost matrix
# cost_matrix = np.array([[0, 10, 15, 20, 25, 30],
#                         [10, 0, 35, 25, 55, 45],
#                         [15, 35, 0, 30, 20, 65],
#                         [20, 25, 30, 0, 55, 40],
#                         [25, 55, 20, 55, 0, 50],
#                         [30, 45, 65, 40, 50, 0]])
#
# # initialize the temperature
# T = 1000
#
# # initialize the number of iterations
# iterations = 0
#
# # initialize the best solution
# best_solution = []
#
# # initialize the current solution
# current_solution = np.random.permutation(6)
#
# # calculate the cost of the current solution
# current_cost = np.sum([cost_matrix[current_solution[i % 6], current_solution[(i + 1) % 6]] for i in range(6)])
#
# # initialize the best cost
# best_cost = current_cost
#
# # iterate while the temperature is greater than 0
# while T > 0:
#
#     # increment the number of iterations
#     iterations += 1
#
#     # choose a random node to swap
#     i = np.random.randint(0, 5)
#     j = np.random.randint(0, 5)
#
#     # swap the nodes
#     current_solution[i], current_solution[j] = current_solution[j], current_solution[i]
#
#     # calculate the cost of the new solution
#     new_cost = np.sum([cost_matrix[current_solution[i % 6], current_solution[(i + 1) % 6]] for i in range(6)])
#
#     # calculate the difference in cost
#     delta_cost = new_cost - current_cost
#
#     # if the new solution is better, accept it
#     if delta_cost <= 0:
#         current_cost = new_cost
#         # if the new solution is better than the best, update the best
#         if new_cost < best_cost:
#             best_cost = new_cost
#             best_solution = current_solution
#     # if the new solution is worse, accept it with a probability equal to e^(-delta_cost/T)
#     else:
#         try:
#             dupa = np.round(np.exp(-delta_cost / T), 2)
#             if random.random() < dupa:
#                 current_cost = new_cost
#         except:
#             print("DUPAAAAA")
#             print(dupa)
#     # decrease the temperature
#     T = T * 0.999
#
# # print the best solution and its cost
# print('Best solution: ', best_solution)
# print('Cost of best solution: ', best_cost)
# print('Number of iterations: ', iterations)


# import random
#
#
# def simulated_annealing(nodes):
#     # initializing variables
#     current_state = random.sample(nodes, len(nodes))
#     temp = 1000
#     alpha = 0.999
#     best_solution = current_state
#     best_cost = calculate_cost(best_solution)
#     while temp > 0.0001:
#         # generating a random neighbor
#         random_index = random.randint(0, len(nodes) - 1)
#         new_state = current_state[:]
#         new_state[random_index], new_state[random_index - 1] = new_state[random_index - 1], new_state[random_index]
#
#         # calculating cost of new state
#         new_cost = calculate_cost(new_state)
#
#         # calculating acceptance probability
#         ap = acceptance_probability(best_cost, new_cost, temp)
#
#         # accepting or rejecting new state
#         if ap > random.random():
#             current_state = new_state
#             best_cost = new_cost
#             if new_cost < best_cost:
#                 best_solution = new_state
#         # cooling down
#         temp = temp * alpha
#     return best_solution
#
#
# # calculate cost of a given solution
# def calculate_cost(solution):
#     cost = 0
#     for i in range(len(solution) - 1):
#         cost += nodes[solution[i]][solution[i + 1]]
#     return cost
#
#
# # calculate acceptance probability
# def acceptance_probability(best_cost, new_cost, temp):
#     if new_cost < best_cost:
#         return 1
#     else:
#         return math.exp((best_cost - new_cost) / temp)
#
#
# # sample nodes
# nodes = {0: {1: 10, 2: 20, 3: 30, 4: 40, 5: 50},
#          1: {0: 10, 2: 25, 3: 35, 4: 45, 5: 55},
#          2: {0: 20, 1: 25, 3: 15, 4: 25, 5: 35},
#          3: {0: 30, 1: 35, 2: 15, 4: 30, 5: 40},
#          4: {0: 40, 1: 45, 2: 25, 3: 30, 5: 10},
#          5: {0: 50, 1: 55, 2: 35, 3: 40, 4: 10}}
#
# # run simulated annealing
# best_solution = simulated_annealing(list(nodes.keys()))
# print('Best solution: ', best_solution)
# print('Cost: ', calculate_cost(best_solution))


# import random
#
#
# def distance(A, B):
#     return ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
#
#
# def cost(solution, cities):
#     total_distance = 0
#     num_cities = len(cities)
#     for i in range(num_cities):
#         j = (i + 1) % num_cities
#         total_distance += distance(cities[solution[i]], cities[solution[j]])
#     return total_distance
#
#
# def random_neighbor(solution):
#     i = random.randint(0, len(solution) - 1)
#     j = random.randint(0, len(solution) - 1)
#     neighbor = solution[:]
#     neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
#     return neighbor
#
#
# def acceptance_probability(old_cost, new_cost, temperature):
#     if new_cost < old_cost:
#         return 1.0
#     else:
#         return (2.718) ** ((old_cost - new_cost) / temperature)
#
#
# def simulated_annealing(cities, max_temp, min_temp, cooling_rate, max_iter):
#     curr_solution = random.sample(range(len(cities)), len(cities))
#     curr_cost = cost(curr_solution, cities)
#     best_solution = curr_solution
#     best_cost = curr_cost
#     temperature = max_temp
#     while temperature > min_temp and max_iter > 0:
#         max_iter -= 1
#         new_solution = random_neighbor(curr_solution)
#         new_cost = cost(new_solution, cities)
#         acceptance = acceptance_probability(curr_cost, new_cost, temperature)
#         if acceptance > random.random():
#             curr_solution = new_solution
#             curr_cost = new_cost
#             if curr_cost < best_cost:
#                 best_solution = curr_solution
#                 best_cost = curr_cost
#         temperature *= 1 - cooling_rate
#     return best_solution, best_cost
#
#
# def main():
#     cities = [(0, 0), (1, 5), (4, 3), (5, 7), (7, 2), (8, 2)]
#     max_temp = 1000
#     min_temp = 0.01
#     cooling_rate = 0.001
#     max_iter = 10000
#     best_solution, best_cost = simulated_annealing(cities, max_temp, min_temp, cooling_rate, max_iter)
#     print(best_solution, best_cost)
#
#
# if __name__ == "__main__":
#     main()


# import numpy as np
# import time
#
# #nodes
# node_X = [0, 2, 4, 6, 8, 10]
# node_Y = [0, 6, 2, 8, 4, 10]
#
# #distance between two nodes
# def distance(node1, node2):
#     dist = np.sqrt((node_X[node1] - node_X[node2])**2 + (node_Y[node1] - node_Y[node2])**2)
#     return dist
#
# #calculate the total distance of the route
# def route_distance(route):
#     total_distance = 0
#     for i in range(len(route)):
#         total_distance += distance(route[i], route[(i + 1) % len(route)])
#     return total_distance
#
# #randomize the initial route
# def random_route(n):
#     route = np.random.permutation(n)
#     return route
#
# #simulated annealing
# def simulated_anneal(route):
#     temp = 10000
#     delta_temp = 0.99
#     min_temp = 0.00001
#     best_distance = route_distance(route)
#     best_route = route
#     while temp > min_temp:
#         i = np.random.randint(0, len(route))
#         j = np.random.randint(0, len(route))
#         new_route = np.copy(route)
#         new_route[i], new_route[j] = new_route[j], new_route[i]
#         current_distance = route_distance(new_route)
#         if current_distance < best_distance:
#             best_distance = current_distance
#             best_route = np.copy(new_route)
#             route = np.copy(new_route)
#         else:
#             prob = np.exp(-(current_distance - best_distance)/temp)
#             if np.random.rand() < prob:
#                 route = np.copy(new_route)
#         temp *= delta_temp
#     return best_route, best_distance
#
# #main
# start = time.time()
# n = 6
# route = random_route(n)
# print("Initial route:", route)
# print("Initial distance:", route_distance(route))
# best_route, best_distance = simulated_anneal(route)
# print("Best route:", best_route)
# print("Best distance:", best_distance)
# end = time.time()
# print("Time:", end - start)


# cos za proste
#
#
# import random
#
# #Function to calculate the cost of the current path
# def cost(path):
#     cost = 0
#     for i in range(len(path)):
#         j = (i+1) % len(path)
#         cost += distances[path[i]][path[j]]
#     return cost
#
# #Simulated annealing algorithm to solve the traveling salesman problem
# def anneal(start_path):
#     temperature = 1000
#     alpha = 0.999
#     while temperature > 0.0001:
#         #Create a new path by making a small change to the existing path
#         new_path = list(start_path)
#         i, j = sorted(random.sample(range(len(start_path)), 2))
#         new_path[i], new_path[j] = start_path[j], start_path[i]
#
#         #Calculate the cost of the new path
#         new_cost = cost(new_path)
#
#         #Calculate the difference in cost between the new path and the current path
#         diff = new_cost - cost(start_path)
#
#         #If the new path is better than the current path, accept it
#         if diff <= 0 or random.random() < math.exp(-diff/temperature):
#             start_path = new_path
#
#         #Decrease the temperature
#         temperature *= alpha
#     return start_path
#
# #Matrix of distances between nodes
# distances = [[0, 12, 3, 23, 1, 5],
#              [12, 0, 9, 18, 3, 41],
#              [3, 9, 0, 89, 56, 21],
#              [23, 18, 89, 0, 87, 3],
#              [1, 3, 56, 87, 0, 55],
#              [5, 41, 21, 3, 55, 0]]
#
# #Randomly generate a starting path
# start_path = list(range(len(distances)))
# random.shuffle(start_path)
#
# #Run the simulated annealing algorithm
# print(anneal(start_path))


import random
import numpy as np

# Matrix of distances between nodes
distances = np.array([
    [0, 10, 15, 20, 30, 40],
    [10, 0, 35, 25, 15, 25],
    [15, 35, 0, 55, 45, 20],
    [20, 25, 55, 0, 30, 20],
    [30, 15, 45, 30, 0, 55],
    [40, 25, 20, 20, 55, 0]
])


# Function to calculate the total distance of the path
def calculateDistance(solution):
    dist = 0
    for i in range(len(solution)):
        if i != len(solution) - 1:
            dist += distances[solution[i], solution[i + 1]]
        else:
            dist += distances[solution[i], solution[0]]
    return dist


# Function to generate a random solution
def randomSolution():
    solution = [i for i in range(len(distances))]
    random.shuffle(solution)
    return solution


# Function to generate a neighbour by switching the position of two randomly selected cities
def neighbour(solution):
    neighbour = solution[:]
    x, y = random.sample(range(len(neighbour)), 2)
    neighbour[x], neighbour[y] = neighbour[y], neighbour[x]
    return neighbour


# Function to run the simulation
def simulatedAnnealing(temp, coolingRate, iterations):
    solution = randomSolution()
    bestSolution = solution[:]
    for i in range(iterations):
        newSolution = neighbour(solution)
        deltaE = calculateDistance(newSolution) - calculateDistance(solution)
        if deltaE < 0 or random.random() < np.exp(-deltaE / temp):
            solution = newSolution[:]
        if calculateDistance(solution) < calculateDistance(bestSolution):
            bestSolution = solution[:]
        temp *= 1 - coolingRate
    return bestSolution


bestSolution = simulatedAnnealing(1000, 0.001, 5000)
print("Best solution: ", bestSolution)
print("Distance: ", calculateDistance(bestSolution))
