import abc
from dataclasses import dataclass
import pandas as pd
import os


@dataclass
class AbstractReport(abc.ABC):
    output_file: str
    instance: str = None
    extra_data: dict = None

    def __post_init__(self):
        self._first_call = True
        self._file_name = None
        self.extra_data = self.extra_data or {}

    def _serialize_params(self, brkga_obj):
        params = self._get_params(brkga_obj)
        params.update(self.extra_data)

        if "criteria" in params:
            del params["criteria"]

        keys = sorted(params.keys())
        keys.remove("instance")
        keys.insert(0, "instance")
        values = [params[p] for p in keys]
        strings = [f"{key}={value}" for key, value in zip(keys, values)]
        return "__".join(strings)

    def _get_file_name(self, brkga_obj, extension):
        if self._file_name is None:
            if os.path.isdir(self.output_file):
                file_name = self._serialize_params(brkga_obj)
                self._file_name = os.path.join(self.output_file, file_name + extension)
            else:
                self._file_name = self.output_file

        return self._file_name

    def _get_params(self, brkga_obj):
        return {
            "instance": self.instance,
            "pop_size": brkga_obj.pop_size,
            "elite_size": brkga_obj.elite_size / brkga_obj.pop_size,
            "mutant_size": brkga_obj.mutant_size / brkga_obj.pop_size,
            "criteria": brkga_obj.criteria,
            "pho_e": brkga_obj.rho_e,
            "k": brkga_obj.k,
            "x_intvl": brkga_obj.x_intvl,
            "t": brkga_obj.t,
            "initial_seed": brkga_obj.initial_seed,
            "epsilon": brkga_obj.epsilon,
            "maximize": brkga_obj.maximize,
            "n": brkga_obj.n,
            **brkga_obj.get_specific_header_params(),
        }

    @abc.abstractmethod
    def make(self, *args, **kwargs):
        raise NotImplementedError()


class PlainTextReport(AbstractReport):
    def make(self, brkga_obj, *args, **kwargs):
        header = self._get_params(brkga_obj)

        mode = "w" if self._first_call else "a"

        file_name = self._get_file_name(brkga_obj, ".txt")

        with open(file_name, mode) as file:
            if self._first_call:
                max_param_name = len(max(header.keys(), key=len)) + 1
                file.write("-------- Params:\n")

                for h, v in header.items():
                    file.write(f"{h.ljust(max_param_name)}:{v}\n")

                file.write("\n-------- Best Individual:\n")
                file.write(f">> Genes: {brkga_obj.best_ind.genes}\n")
                file.write(f">> Solution: {brkga_obj.best_ind.solution}\n")
                file.write(f">> Fitness: {brkga_obj.best_ind.fitness}\n")
                file.write(f">> Generation: {brkga_obj.best_ind.generation}\n")

                file.write("\n-------- Data:\n")
                self._first_call = False

            max_param_name = (
                len(max(list(brkga_obj.output.items())[0][1].keys(), key=len)) + 1
            )

            for g, data in brkga_obj.output.items():
                file.write(f"Generation: {g}\n")
                for result, value in data.items():
                    file.write(f">> {result.ljust(max_param_name)}:{value}\n")


@dataclass
class CsvReport(AbstractReport):
    mode: str = "w"

    def make(self, brkga_obj, *args, **kwargs):
        file_name = self._get_file_name(brkga_obj, ".csv")
        header = self._get_params(brkga_obj)
        output = brkga_obj.output
        data = [
            {"generation": gen, **data, **header, **self.extra_data}
            for gen, data in output.items()
        ]

        df = pd.DataFrame(data)

        if self._first_call:
            df.to_csv(file_name, mode=self.mode, index=False)
            self._first_call = False
        else:
            df.to_csv(file_name, mode="a", index=False, header=False)
