import numpy as np

class HillClimbing:
    def __init__(self, fitness_func, lower_bound, upper_bound, step_size=0.1, max_iterations=100, max_nfe=10000, d=2):
        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace(
            '_function', '').title()
        self.dim = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.max_nfe = max_nfe
        self.nfe = 0
        self.current_solution = None
        self.current_fitness = float('inf')
        self.positions_history = []
        self.overall_best_solution = None
        self.overall_best_fitness = float('inf')
        self.history_best_fitness = []

    def _initialize(self):

        self.current_solution = np.random.uniform(
            low=self.lower_bound, high=self.upper_bound, size=self.dim
        )
        self.current_fitness = self.fitness_func(self.current_solution)
        self.nfe += 1
        self.overall_best_solution = self.current_solution
        self.overall_best_fitness = self.current_fitness

    def _get_neighbor(self):

        neighbor = np.copy(self.current_solution)

        perturbation = np.random.uniform(-self.step_size,
                                         self.step_size, size=self.dim)
        neighbor += perturbation

        neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
        return neighbor

    def run_by_iteration(self):

        self._initialize()
        self.history_best_fitness = []
        self.positions_history = []

        for _ in range(self.max_iterations):
            self.positions_history.append(np.array([self.current_solution]))

            neighbor = self._get_neighbor()
            neighbor_fitness = self.fitness_func(neighbor)
            self.nfe += 1

            if neighbor_fitness < self.current_fitness:
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness

                if self.current_fitness < self.overall_best_fitness:
                    self.overall_best_fitness = self.current_fitness
                    self.overall_best_solution = self.current_solution

            self.history_best_fitness.append(self.overall_best_fitness)

        self.positions_history.append(np.array([self.current_solution]))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

    def run_by_nfe(self):

        self._initialize()
        self.history_best_fitness = []
        self.positions_history = []

        while self.nfe < self.max_nfe:
            self.positions_history.append(np.array([self.current_solution]))

            neighbor = self._get_neighbor()
            neighbor_fitness = self.fitness_func(neighbor)
            self.nfe += 1

            if self.nfe >= self.max_nfe:
                break

            if neighbor_fitness < self.current_fitness:
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness
                if self.current_fitness < self.overall_best_fitness:
                    self.overall_best_fitness = self.current_fitness
                    self.overall_best_solution = self.current_solution
            self.history_best_fitness.append(self.overall_best_fitness)
        self.positions_history.append(np.array([self.current_solution]))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness


class SimulatedAnnealing:

    def __init__(self, fitness_func, lower_bound, upper_bound, initial_temp=100.0, cooling_rate=0.99, step_size=0.1, max_iterations=100, max_nfe=10000, d=2):
        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace(
            '_function', '').title()
        self.dim = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.max_nfe = max_nfe
        self.nfe = 0
        self._current_iteration = 0
        self.current_solution = None
        self.current_fitness = float('inf')
        self.positions_history = []
        self.overall_best_solution = None
        self.overall_best_fitness = float('inf')
        self.history_best_fitness = []

    def _initialize(self):

        self.current_solution = np.random.uniform(
            low=self.lower_bound, high=self.upper_bound, size=self.dim
        )
        self.current_fitness = self.fitness_func(self.current_solution)
        self.nfe += 1
        self.overall_best_solution = self.current_solution.copy()
        self.overall_best_fitness = self.current_fitness

    def _get_neighbor(self):

        neighbor = np.copy(self.current_solution)

        perturbation = np.random.uniform(-self.step_size,
                                         self.step_size, size=self.dim)
        neighbor += perturbation

        neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
        return neighbor

    def _acceptance_probability(self, new_fitness, iteration):

        current_T = self.initial_temp * (self.cooling_rate ** iteration)

        if new_fitness < self.current_fitness:
            return 1.0

        delta_E = new_fitness - self.current_fitness

        if current_T > 0:
            return np.exp(-delta_E / current_T)
        else:
            return 0.0

    def run_by_iteration(self):

        self._initialize()
        self.history_best_fitness = []
        self.positions_history = []

        for iteration in range(self.max_iterations):
            self.positions_history.append(np.array([self.current_solution]))

            neighbor = self._get_neighbor()
            neighbor_fitness = self.fitness_func(neighbor)
            self.nfe += 1

            prob = self._acceptance_probability(neighbor_fitness, iteration)

            if np.random.random() < prob:
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness

                if self.current_fitness < self.overall_best_fitness:
                    self.overall_best_fitness = self.current_fitness
                    self.overall_best_solution = self.current_solution.copy()

            self.history_best_fitness.append(self.overall_best_fitness)

        self.positions_history.append(np.array([self.current_solution]))

        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

    def run_by_nfe(self):

        self._initialize()
        self.history_best_fitness = []
        self.positions_history = []

        iteration = 0
        while self.nfe < self.max_nfe:
            self.positions_history.append(np.array([self.current_solution]))

            neighbor = self._get_neighbor()
            neighbor_fitness = self.fitness_func(neighbor)
            self.nfe += 1

            if self.nfe >= self.max_nfe:
                break

            prob = self._acceptance_probability(neighbor_fitness, iteration)

            if np.random.random() < prob:
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness
                if self.current_fitness < self.overall_best_fitness:
                    self.overall_best_fitness = self.current_fitness
                    self.overall_best_solution = self.current_solution.copy()
            self.history_best_fitness.append(self.overall_best_fitness)
            iteration += 1

        self.positions_history.append(np.array([self.current_solution]))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness


class GeneticAlgorithm:

    def __init__(self, fitness_func, lower_bound, upper_bound, population_size=100, max_iterations=100, max_nfe=10000, crossover_rate=0.8, mutation_rate=0.1, d=2):

        self.fitness_func = fitness_func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.pop_size = population_size
        self.max_iterations = max_iterations
        self.max_nfe = max_nfe
        self.nfe = 0
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.dim = d
        self.function_name = fitness_func.__name__.replace(
            '_function', '').title()

        self.population = None
        self.fitness_values = None
        self.positions_history = []
        self.overall_best_solution = None
        self.overall_best_fitness = float('inf')
        self.history_best_fitness = []

    def _initialize_population(self):

        self.population = np.random.uniform(
            low=self.lower_bound,
            high=self.upper_bound,
            size=(self.pop_size, self.dim)
        )

        self.fitness_values = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            self.fitness_values[i] = self.fitness_func(self.population[i])
            self.nfe += 1

        self.overall_best_fitness = np.min(self.fitness_values)
        self.overall_best_solution = self.population[
            np.argmin(self.fitness_values)
        ]

    def _selection(self):

        tournament_size = 3
        selected_parents = []
        for _ in range(self.pop_size):

            competitor_indices = np.random.choice(
                self.pop_size,
                size=tournament_size,
                replace=False
            )

            best_competitor_index = competitor_indices[
                np.argmin(self.fitness_values[competitor_indices])
            ]
            selected_parents.append(self.population[best_competitor_index])
        return np.array(selected_parents)

    def _crossover(self, parent1, parent2):

        if np.random.random() < self.crossover_rate:
            alpha = 0.5
            child1 = np.empty(self.dim)
            child2 = np.empty(self.dim)
            for d in range(self.dim):

                d_val = abs(parent1[d] - parent2[d])
                min_val = min(parent1[d], parent2[d]) - alpha * d_val
                max_val = max(parent1[d], parent2[d]) + alpha * d_val

                child1[d] = np.random.uniform(min_val, max_val)
                child2[d] = np.random.uniform(min_val, max_val)
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def _mutation(self, individual):

        mutated_individual = np.copy(individual)
        for d in range(self.dim):
            if np.random.random() < self.mutation_rate:
                sigma = 0.2 * (self.upper_bound - self.lower_bound) / 10
                perturbation = np.random.normal(0, sigma)
                mutated_individual[d] += perturbation
        mutated_individual = np.clip(
            mutated_individual, self.lower_bound, self.upper_bound
        )
        return mutated_individual

    def _evaluate_population(self):

        self.fitness_values = np.zeros(self.pop_size)
        for i in range(self.pop_size):

            if self.nfe >= self.max_nfe and self.max_nfe > 0:

                self.fitness_values[i] = float('inf')
            else:
                self.fitness_values[i] = self.fitness_func(self.population[i])
                self.nfe += 1

        current_best_fitness = np.min(self.fitness_values)
        if current_best_fitness < self.overall_best_fitness:
            self.overall_best_fitness = current_best_fitness
            self.overall_best_solution = self.population[
                np.argmin(self.fitness_values)
            ]

    def _run_one_generation(self):

        parents = self._selection()
        next_population = []

        for i in range(0, self.pop_size, 2):
            p1 = parents[i]
            p2 = parents[i + 1] if i + 1 < self.pop_size else parents[0]
            child1, child2 = self._crossover(p1, p2)
            child1 = self._mutation(child1)
            child2 = self._mutation(child2)
            next_population.extend([child1, child2])

        self.population = np.array(next_population[:self.pop_size])

        self._evaluate_population()

    def run_by_iteration(self):

        self._initialize_population()
        self.history_best_fitness = []
        self.positions_history = []

        for _ in range(self.max_iterations):
            self.positions_history.append(np.copy(self.population))
            self._run_one_generation()
            self.history_best_fitness.append(self.overall_best_fitness)

        self.positions_history.append(np.copy(self.population))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

    def run_by_nfe(self):

        self._initialize_population()
        self.history_best_fitness = []
        self.positions_history = []

        for _ in range(self.max_iterations):
            if self.nfe >= self.max_nfe:
                break

            self.positions_history.append(np.copy(self.population))
            self._run_one_generation()
            self.history_best_fitness.append(self.overall_best_fitness)

        self.positions_history.append(np.copy(self.population))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness
