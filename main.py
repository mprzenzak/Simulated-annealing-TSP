import os
import sys
import time

from src.data_controller import read_matrix, load_config
from src.calculator import run_test
from src.logger import Logger


def main(file, optimal_solution, temp, cooling_rate, cooling_type, eras_number, era_length, solution_in_neighbourhood,
         cfg: dict = None):
    if file:
        cfg = {'Temperature': temp,
               'Minimal_temperature': 0.0001,
               'Cooling_Rate': cooling_rate,
               'Eras': eras_number,
               'Era_length': era_length,
               'Solution_in_neighbourhood': solution_in_neighbourhood,
               'Cooling_Type': cooling_type,
               'Data': file,
               'Results': 'results.csv',
               'Precision': 4,
               'Solution': optimal_solution,
               'Repeats': 3}

        output = run_solve(cfg)
        save_to_file(cfg, output)
    else:
        if cfg is None:
            cfg = load_config()

        output = run_solve(cfg)
        save_to_file(cfg, output)


def run_solve(cfg: dict):
    print('[OK] Rozwiazywanie\n')

    repeats = cfg['Repeats']
    matrix = read_matrix(f'{"./input"}/{cfg["Data"]}')

    outputs = []
    for i in range(repeats):
        if i > 1: print(f'[{i + 1}/{repeats}]      ', end='\r')
        outputs.append(run_test(matrix, cfg))

    final_output = get_output_struct(cfg)

    for output in outputs:
        final_output['time'] += output['time']
        final_output['memory_usage'] += output['memory_usage']

        solution = output['solution']
        if solution < final_output['solution']:
            final_output['solution'] = solution
            final_output['path'] = output['path']

    final_output['accuracy'] = round(cfg["Solution"] / final_output["solution"] * 100)
    final_output['time'] = str(round(final_output['time'] / repeats / 1_000_000, cfg['Precision'])).replace('.', ',')
    final_output['memory_usage'] = str(round(final_output['memory_usage'] / repeats, cfg['Precision'])).replace('.',
                                                                                                                ',')

    print_results(final_output, cfg)
    return final_output


def get_output_struct(cfg):
    return {
        'input_file': cfg['Data'],
        'repeats': cfg['Repeats'],
        'solution': float('inf'),
        'accuracy': 0,
        'path': (),
        'temp': cfg['Temperature'],
        'cooling_rate': cfg['Cooling_Rate'],
        'solution_in_neighbourhood': cfg['Solution_in_neighbourhood'],
        'cooling_type': cfg['Cooling_Type'],
        'eras': cfg['Eras'],
        'length_of_era': cfg['Era_length'],
        'time': 0,
        'memory_usage': 0
    }


def print_results(final_output: dict, cfg: dict):
    print('Rozwiazanie: ', final_output['solution'])
    print('Sciezka: ', final_output['path'])

    if cfg['Solution'] is not None:
        print(f'\nDokladnosc: {final_output["accuracy"]}%')
    print(f'Czas: {final_output["time"]}ms')
    print(f'Uzycie pamieci: {final_output["memory_usage"]}MB')


def save_to_file(cfg: dict, output: dict):
    logger = Logger('./output', cfg['Results'])
    logger.log(output)


if __name__ == '__main__':
    test_list = []
    if os.path.exists("output/results.csv"):
        os.remove("output/results.csv")

    temp, cooling_rate, cooling_type, eras_number, era_length, solution_in_neighbourhood = None, None, None, None, None, None

    # data_files_list = ["tsp_6_1.txt", "tsp_6_2.txt", "tsp_10.txt", "tsp_12.txt", "tsp_13.txt", "tsp_14.txt"]
    data_files_list = ["tsp_17.txt", "bays29.tsp", "rat99.tsp", "si175.tsp"]
    # optimal_solutions_list = [132, 80, 212, 264, 269, 282]
    optimal_solutions_list = [39, 2020, 1211, 21407]

    temperatures_list = [40, 100, 1000]
    cooling_rate_list = [0.90, 0.94, 0.99]
    eras_number_list = [200, 500, 10000]
    era_length_list = [1, 2, 50]
    solutions_in_neighbourhood_list = ["2swaps", "arc"]
    cooling_type_list = ["Geometrical", "Boltzmann"]

    # for i in range(len(data_files_list)):
    #     for temp in temperatures_list:
    #         for cooling_type in cooling_type_list:
    #             for eras_number in eras_number_list:
    #                 for era_length in era_length_list:
    #                     for solution_in_neighbourhood in solutions_in_neighbourhood_list:
    #                         for cooling_rate in cooling_rate_list:
    #                             main(data_files_list[i], optimal_solutions_list[i], temp, cooling_rate,
    #                                  cooling_type, eras_number, era_length, solution_in_neighbourhood)
    # sys.exit('\n[OK] Done')

    main(None, None, None, None, None, None, None, None)
    print("Koniec")
    time.sleep(10000)
    # sys.exit('\n[OK] Done')
