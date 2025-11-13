import numpy as np

class ArtificialBeeColony:
    def __init__(self, fitness_func, lower_bound, upper_bound, num_employed=50, num_onlooker=50, limit=10, max_iterations=100, max_nfe=10000, d=2):

        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace(
            '_function', '').title()
        self.dim = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_employed_bees = num_employed
        self.num_onlooker_bees = num_onlooker
        self.limit = limit
        self.max_iterations = max_iterations
        self.max_nfe = max_nfe
        self.nfe = 0
        self.food_sources = None
        self.fitness_values = None
        self.no_improvement_counters = None
        self.positions_history = []
        self.overall_best_solution = None
        self.overall_best_fitness = float('inf')
        self.history_best_fitness = []

    def _initialize_population(self):

        self.food_sources = np.random.uniform(
            low=self.lower_bound, high=self.upper_bound,
            size=(self.num_employed_bees, self.dim)
        )
        self.fitness_values = np.zeros(self.num_employed_bees)
        for i in range(self.num_employed_bees):
            self.fitness_values[i] = self.fitness_func(self.food_sources[i])
            self.nfe += 1
        self.no_improvement_counters = np.zeros(self.num_employed_bees)
        self.overall_best_fitness = np.min(self.fitness_values)
        self.overall_best_solution = self.food_sources[np.argmin(
            self.fitness_values)]

    def _update_food_source(self, current_index, k_index):

        j = np.random.randint(0, self.dim)
        mutant = np.copy(self.food_sources[current_index])

        phi = np.random.uniform(-1, 1)

        mutant[j] = self.food_sources[current_index][j] + phi * \
            (self.food_sources[current_index]
             [j] - self.food_sources[k_index][j])

        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

        mutant_fitness = self.fitness_func(mutant)
        self.nfe += 1

        if mutant_fitness < self.fitness_values[current_index]:
            self.food_sources[current_index] = mutant
            self.fitness_values[current_index] = mutant_fitness
            self.no_improvement_counters[current_index] = 0

            if mutant_fitness < self.overall_best_fitness:
                self.overall_best_fitness = mutant_fitness
                self.overall_best_solution = mutant
        else:
            self.no_improvement_counters[current_index] += 1

    def _employed_bees_phase(self):

        for i in range(self.num_employed_bees):
            k = i

            while k == i:
                k = np.random.randint(0, self.num_employed_bees)
            self._update_food_source(i, k)

    def _onlooker_bees_phase(self):

        qualities = 1.0 / (1.0 + self.fitness_values)
        total_quality = np.sum(qualities)

        if total_quality == 0:
            probabilities = np.ones(
                self.num_employed_bees) / self.num_employed_bees
        else:
            probabilities = qualities / total_quality

        for _ in range(self.num_onlooker_bees):
            selected_index = np.random.choice(
                self.num_employed_bees, p=probabilities)
            k = selected_index

            while k == selected_index:
                k = np.random.randint(0, self.num_employed_bees)
            self._update_food_source(selected_index, k)

    def _scout_bees_phase(self):

        for k_scout in range(self.num_employed_bees):

            if self.no_improvement_counters[k_scout] > self.limit:
                new_source = np.random.uniform(
                    low=self.lower_bound, high=self.upper_bound, size=self.dim
                )
                self.food_sources[k_scout] = new_source
                self.fitness_values[k_scout] = self.fitness_func(new_source)
                self.nfe += 1
                self.no_improvement_counters[k_scout] = 0

                if self.fitness_values[k_scout] < self.overall_best_fitness:
                    self.overall_best_fitness = self.fitness_values[k_scout]
                    self.overall_best_solution = new_source
                break

    def run_by_iteration(self):

        self._initialize_population()
        self.history_best_fitness = []
        self.positions_history = []

        for _ in range(self.max_iterations):
            self.positions_history.append(np.copy(self.food_sources))
            self._employed_bees_phase()
            self._onlooker_bees_phase()
            self._scout_bees_phase()
            self.history_best_fitness.append(self.overall_best_fitness)
        self.positions_history.append(np.copy(self.food_sources))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

    def run_by_nfe(self):

        self._initialize_population()
        self.history_best_fitness = []
        self.positions_history = []
        while self.nfe < self.max_nfe:
            self.positions_history.append(np.copy(self.food_sources))
            self._employed_bees_phase()
            if self.nfe >= self.max_nfe:
                break
            self._onlooker_bees_phase()
            if self.nfe >= self.max_nfe:
                break
            self._scout_bees_phase()
            self.history_best_fitness.append(self.overall_best_fitness)
        self.positions_history.append(np.copy(self.food_sources))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness
