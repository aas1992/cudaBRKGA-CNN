import abc
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Union, List

from brkga.brkga import Individual
from brkga.single import SingleObjectiveBRKGA


@dataclass
class AsPopulationArray:
    population: Any = None
    pop_fitness: Any = None

    def __post_init__(self):
        if self.pop_fitness is None:
            if self.population is not None:
                self.pop_fitness = np.full(
                    (self.population.shape[0],), -1.00, dtype=float
                )

    def __getitem__(self, item):
        population = self.population[item]
        fitness = self.pop_fitness[item]
        return AsPopulationArray(
            population=population, pop_fitness=fitness
        )

    def join(self, other_pop):
        new_pop = np.concatenate((self.population, other_pop.population), 0)
        new_fitness = np.concatenate((self.pop_fitness, other_pop.pop_fitness), 0)
        return AsPopulationArray(
            population=new_pop, pop_fitness=new_fitness
        )


class DefaultBRKGA(SingleObjectiveBRKGA, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population: Union[
            List[AsPopulationArray], List[None]] = self.k * [None]

    def create_population(self, qtd) -> AsPopulationArray:
        population = np.random.rand(qtd, self.n)
        return AsPopulationArray(population=population)

    @abc.abstractmethod
    def decoder(self, genes, **kwargs):
        """
            Decoder: Function responsible for transforming the vector of
            random keys into a solution for the specific problem.
            It maps a solution to the problem from the vector.
        """
        raise NotImplementedError("Method not implemented!")

    @abc.abstractmethod
    def calculate_fitness(self, solution, *fitness_args):
        """
            Fitness: Function responsible for evaluating the quality of the
            individual's solution according to the specific problem.
        """
        raise NotImplementedError("Method not implemented!")

    def sort_population(self) -> List[AsPopulationArray]:
        if self.maximize:
            for i in range(self.k):
                arg = np.argsort(self.population[i].pop_fitness)
                self.population[i] = self.population[i][arg[::-1]]
        else:
            for i in range(self.k):
                arg = np.argsort(self.population[i].pop_fitness)
                self.population[i] = self.population[i][arg]
        return self.population

    def get_best_ind_from_population(self) -> Individual:
        if self.maximize:
            all_best_index = [np.argmax(p.pop_fitness) for p in self.population]
            fn_compare = max
        else:
            all_best_index = [np.argmin(p.pop_fitness) for p in self.population]
            fn_compare = min

        all_best_fitness = [self.population[i].pop_fitness[best_idx]
                            for i, best_idx in enumerate(all_best_index)]

        zipped = zip(range(self.k), all_best_index, all_best_fitness)

        best_pop, best_idx, best_fitness = fn_compare(zipped, key=lambda p: p[-1])

        genes = self.population[best_pop].population[best_idx, :]
        return Individual(
            genes=genes, fitness=best_fitness, solution=self.decoder(genes)
        )

    def calculate_all_fitness(self, population: AsPopulationArray):
        for i, genes in enumerate(population.population):
            solution = self.decoder(genes)
            population.pop_fitness[i] = self.calculate_fitness(solution)

    def make_crossover(self, parent1, parent2):
        # offs_len X qtd_genes (floats)
        probs = np.random.rand(parent1.shape[0], parent1.shape[-1])
        offs = np.where(probs <= self.get_rho_e(), parent1, parent2)
        return offs

    def create_offsprings(self, elite, non_elite, offs_len) -> AsPopulationArray:
        assert elite.population.shape[-1] == non_elite.population.shape[-1]

        pop_elite_choices = np.random.randint(0, elite.population.shape[0], (offs_len,))
        pop_non_elite_choices = np.random.randint(0, non_elite.population.shape[0], (offs_len,))
        p1 = elite.population[pop_elite_choices, :]  # parents1
        p2 = non_elite.population[pop_non_elite_choices, :]  # parents2

        offs = self.make_crossover(p1, p2)
        return AsPopulationArray(population=offs)

    def exchanging_best_chromosomes(self, population: List[AsPopulationArray]) -> List[AsPopulationArray]:
        t = self.t
        for i in range(self.k):
            if i < self.k - 1:
                population[i + 1].population[-t:] = population[i].population[:t]
                population[i + 1].pop_fitness[-t:] = population[i].pop_fitness[:t]
            else:
                population[0].population[-t:] = population[-1].population[:t]
                population[0].pop_fitness[-t:] = population[-1].pop_fitness[:t]

        return population

    def log(self, ind, time_ind):
        self.output[self.generation] = {
            "solution": self.best_ind.solution.tolist(),
            "fitness": self.best_ind.fitness.item(),
            "genes": self.best_ind.genes.tolist(),
            "time": time_ind,
        }

    def run(self):
        start_time = time.time()
        for i in range(self.k):
            self.population[i] = self.create_population(self.pop_size)
            self.calculate_all_fitness(self.population[i])

        self.update_best_ind(self.get_best_ind_from_population())
        previous_fitness = self.best_ind.fitness
        any_criterion_met = False

        while not any_criterion_met:
            self.generation += 1
            print("Generation: ", self.generation, "Best fitness: ", self.best_ind.fitness)

            self.population = self.sort_population()

            if (self.k > 1) and (self.generation % self.x_intvl == 0):
                self.population = self.exchanging_best_chromosomes(self.population)

            for i in range(self.k):
                population = self.population[i]

                elite = population[: self.elite_size]
                non_elite = population[self.elite_size:]
                mutants = self.create_population(self.mutant_size)

                offs_len = self.pop_size - self.elite_size - self.mutant_size
                offsprings = self.create_offsprings(elite, non_elite, offs_len)

                new_pop = mutants.join(offsprings)
                self.calculate_all_fitness(new_pop)
                self.population[i] = elite.join(new_pop)

            if self.maximize:
                [
                    [
                        self.log(self.population[j].population[i], time.time() - start_time)
                        for i in range(len(self.population[j].population))
                        if self.population[j].pop_fitness[i] > self.best_ind.fitness
                    ]
                    for j in range(len(self.population))
                ]
            else:
                [
                    [
                        self.log(self.population[j].population[i], time.time() - start_time)
                        for i in range(len(self.population[j].population))
                        if self.population[j].pop_fitness[i] < self.best_ind.fitness
                    ]
                    for j in range(len(self.population))
                ]

            self.update_best_ind(self.get_best_ind_from_population())

            if not self.has_improvement(previous_fitness):
                self.generations_no_improvement += 1
            else:
                # print(self.best_ind)
                # print("Generation: ", self.generation, "Best fitness: ", self.best_ind.fitness)
                self.log(self.best_ind, time.time() - start_time)
                self.generations_no_improvement = 0
                previous_fitness = self.best_ind.fitness
                self.export_partial_results()

            any_criterion_met = self.criteria.is_met(self)

        self.runtime = time.time() - start_time


if __name__ == "__main__":
    pop = AsPopulationArray(population=np.ones((10, 5)), pop_fitness=np.random.rand(10))
    pop_slice = pop[:5]
    print(pop_slice.population, pop_slice.pop_fitness)
