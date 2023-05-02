import abc
import os
from dataclasses import dataclass
from typing import Any, Optional, Union
from brkga.reports import AbstractReport

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass(repr=True)
class Individual:
    fitness: Any = None
    genes: Any = None
    solution: Any = None
    generation: int = None

    def __eq__(self, other):
        return id(self) == id(other) or all(
            [
                self.fitness == other.fitness,
                tuple(self.genes) == tuple(other.genes),
                tuple(self.solution) == tuple(other.solution),
            ],
        )


@dataclass
class BRKGA(abc.ABC):
    n: int  # Size of each chromosome
    pop_size: int  # Size of each population
    elite_size: float  # Fraction of the population that belongs to the Elite set
    mutant_size: float  # Fraction of the population that receives the Mutants
    criteria: "Criterion"  # Criterion contains a list containing the BRKGA stopping criterion
    # Probability of offspring inheriting an allele from the elite parent
    rho_e: Optional[float] = None
    k: Optional[int] = 1  # Number of independent populations
    # Number of generations to exchange the best individuals between independent populations
    x_intvl: Optional[int] = 0
    t: Optional[Union[int, float]] = 0  # Swap the t best individuals
    initial_seed: Optional[int] = 0  # choosen seed
    # Allowable deviation for solutions close to optimal
    epsilon: Optional[float] = 1e-6
    maximize: Optional[bool] = True
    reporter: AbstractReport = None

    def __post_init__(self):
        self.elite_size = int(round(self.pop_size * self.elite_size))
        self.mutant_size = int(round(self.pop_size * self.mutant_size))

        if isinstance(self.t, float):
            self.t = int(round(self.pop_size * self.t))

        self.best_ind = None  # Save the best individual
        self.generation = 0  # Generation starts at 0
        # Generation without improvement starts at 0
        self.runtime = 0  # Save the running time of the algorithm
        self.generations_no_improvement = 0
        self.output = {}

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError("Method not implemented!")

    def get_rho_e(self):
        return self.rho_e

    def get_specific_header_params(self):
        return {}

    def export_partial_results(self):
        if self.reporter:
            self.reporter.make(self)
            self.output.clear()

    def get_results(self, name_file):
        n_file = str(name_file)
        header = {
            "Instance": n_file,
            "pop_size": self.pop_size,
            "elite_size": self.elite_size / self.pop_size,
            "mutant_size": self.mutant_size / self.pop_size,
            "criteria": self.criteria,
            "pho_e": self.rho_e,
            "k": self.k,
            "x_intvl": self.x_intvl,
            "t": self.t,
            "initial_seed": self.initial_seed,
            "epsilon": self.epsilon,
            "maximize": self.maximize,
            **self.get_specific_header_params(),
        }
        max_param_name = len(max(header.keys(), key=len)) + 1

        with open(f"{n_file}.txt", "w") as file:
            file.write("-------- Params:\n")
            for h, v in header.items():
                file.write(f"{h.ljust(max_param_name)}:{v}\n")

            file.write("\n-------- Best Individual:\n")
            file.write(f">> Genes: {self.best_ind.genes}\n")
            file.write(f">> Solution: {self.best_ind.solution}\n")
            file.write(f">> Fitness: {self.best_ind.fitness}\n")
            file.write(f">> Generation: {self.best_ind.generation}\n")

            file.write("\n-------- Data:\n")
            max_param_name = (
                len(max(list(self.output.items())[0][1].keys(), key=len)) + 1
            )
            for g, data in self.output.items():
                file.write(f"Generation: {g}\n")
                for result, value in data.items():
                    file.write(f">> {result.ljust(max_param_name)}:{value}\n")

            file.close()

    def plot(self, name_file):
        raise NotImplementedError("Method not Implemented")
