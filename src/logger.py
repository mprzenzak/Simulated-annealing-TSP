import os


class Logger:
    def __init__(self, out_dir, file_name):
        self.output_dir = out_dir
        self.output_path = f'{out_dir}/{file_name}'

    def get_fields(self, output: dict):
        return {
            'instancja': output['input_file'],
            'najmniejszy_koszt.': output["solution"],
            'dokladnosc [%]': output["accuracy"],
            'czas [ms]': output["time"],
            'zuzycie_pamieci [MB]': output["memory_usage"],
            'naj_sciezka': output["path"],
            'temperatura_poczatkowa': output["temp"],
            'Sposob_chlodzenia': output["cooling_type"],
            'Wspolczynnik_chlodzenia': output["cooling_rate"],
            'Liczba_epok': output["eras"],
            'Dlugosc_epoki': output["length_of_era"],
            'Przeszukiwanie_sasiedztwa': output["solution_in_neighbourhood"],
        }

    def get_header(self, fields: dict) -> str:
        header = ''
        for field in fields:
            header += f'{field}\t'
        header += '\n'

        return header

    def write_header(self, header: str):
        with open(self.output_path, 'w') as f:
            f.write(header)

    def write_fields(self, fields):
        fields_line = ''
        for field in fields:
            fields_line += f'{fields[field]}\t'
        fields_line += '\n'

        with open(self.output_path, 'a') as f:
            f.write(fields_line)

    def log(self, output: dict):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        fields = self.get_fields(output)
        header = self.get_header(fields)

        if not os.path.exists(self.output_path):
            self.write_header(header)

        self.write_fields(fields)
