import numpy as np
import math


class CuckooSearch:

    def __init__(self, fitness_func, lower_bound, upper_bound, d=2,
                 n=25, pa=0.15, alpha=0.5, max_iterations=100, max_nfe=10000,
                 F=0.5, CR=0.9, levy_beta=1.5, seed=None):

        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace(
            '_function', '').title()
        self.dim = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.n = n
        self.pa = pa
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.max_nfe = max_nfe
        self.nfe = 0
        self.F = F
        self.CR = CR
        self.levy_beta = levy_beta
        self.seed = seed

        self.nests = None
        self.fitness = None

        self.positions_history = []
        self.overall_best_solution = None
        self.overall_best_fitness = float('inf')
        self.history_best_fitness = []

    def _levy_flight(self):

        Lambda = self.levy_beta
        dim = self.dim
        sigma = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
                 (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / Lambda)
        return step

    def _initialize_population(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        self.nests = np.random.uniform(
            self.lower_bound, self.upper_bound, size=(self.n, self.dim))

        self.fitness = np.zeros(self.n)
        for i in range(self.n):
            self.fitness[i] = self.fitness_func(self.nests[i])
            self.nfe += 1

        best_idx = np.argmin(self.fitness)
        self.overall_best_fitness = self.fitness[best_idx]
        self.overall_best_solution = self.nests[best_idx].copy()

    def _update_best(self, new_solution, new_fitness):

        if new_fitness < self.overall_best_fitness:
            self.overall_best_fitness = new_fitness
            self.overall_best_solution = new_solution.copy()

    def _levy_flights_phase(self):

        if self.nfe >= self.max_nfe:
            return

        for i in range(self.n):
            if self.nfe >= self.max_nfe:
                break

            step = self._levy_flight()
            phi = np.random.rand()

            random_nest_idx = np.random.choice(self.nests.shape[0])

            new_nest = self.nests[i] + self.alpha * step * \
                (self.nests[i] - self.nests[random_nest_idx])

            new_nest += phi * (self.overall_best_solution -
                               self.nests[i]) * 0.5
            new_nest = np.clip(new_nest, self.lower_bound, self.upper_bound)

            new_fit = self.fitness_func(new_nest)
            self.nfe += 1

            if new_fit < self.fitness[i]:
                self.nests[i] = new_nest
                self.fitness[i] = new_fit
                self._update_best(new_nest, new_fit)

    def _mixing_phase(self):

        if self.nfe >= self.max_nfe:
            return

        new_nests = self.nests.copy()
        for i in range(self.n):
            idxs = [idx for idx in range(self.n) if idx != i]
            a, b = np.random.choice(idxs, 2, replace=False)

            mutant = self.nests[i] + self.F * (self.nests[a] - self.nests[b])
            cross = np.random.rand(self.dim) < self.CR
            if not np.any(cross):
                cross[np.random.randint(self.dim)] = True
            trial = np.where(cross, mutant, self.nests[i])
            new_nests[i] = np.clip(trial, self.lower_bound, self.upper_bound)

        new_fitness = np.zeros(self.n)
        for i in range(self.n):
            if self.nfe >= self.max_nfe:
                break
            new_fitness[i] = self.fitness_func(new_nests[i])
            self.nfe += 1

        improved = new_fitness < self.fitness
        self.fitness[improved] = new_fitness[improved]
        self.nests[improved] = new_nests[improved]

        best_idx_after_mix = np.argmin(self.fitness)
        if self.fitness[best_idx_after_mix] < self.overall_best_fitness:
            self._update_best(
                self.nests[best_idx_after_mix], self.fitness[best_idx_after_mix])

    def _discovery_phase(self):

        if self.nfe >= self.max_nfe:
            return

        num_replace = max(1, int(self.pa * self.n))
        worst_idx = np.argsort(self.fitness)[-num_replace:]

        for idx in worst_idx:
            if self.nfe >= self.max_nfe:
                break

            if np.random.rand() < 0.5:

                self.nests[idx] = np.random.uniform(
                    self.lower_bound, self.upper_bound, self.dim)
            else:

                self.nests[idx] = np.clip(self.overall_best_solution +
                                          np.random.normal(
                                              0, 0.1 * (self.upper_bound - self.lower_bound), size=self.dim),
                                          self.lower_bound, self.upper_bound)

            self.fitness[idx] = self.fitness_func(self.nests[idx])
            self.nfe += 1

            self._update_best(self.nests[idx], self.fitness[idx])

    def _local_search_phase(self):

        if self.nfe >= self.max_nfe:
            return

        local = self.overall_best_solution + \
            np.random.normal(0, 0.01 * (self.upper_bound -
                             self.lower_bound), size=(5, self.dim))
        local = np.clip(local, self.lower_bound, self.upper_bound)

        local_f = np.zeros(5)
        for i in range(5):
            if self.nfe >= self.max_nfe:
                break
            local_f[i] = self.fitness_func(local[i])
            self.nfe += 1

        mi = np.argmin(local_f)
        if local_f[mi] < self.overall_best_fitness:
            self.overall_best_solution = local[mi].copy()
            self.overall_best_fitness = local_f[mi]

            worst = np.argmax(self.fitness)
            self.nests[worst] = local[mi].copy()
            self.fitness[worst] = local_f[mi]

    def run_by_iteration(self):

        self._initialize_population()
        self.history_best_fitness = []
        self.positions_history = []

        for i in range(self.max_iterations):
            self.positions_history.append(np.copy(self.nests))

            self._levy_flights_phase()
            self._mixing_phase()
            self._discovery_phase()

            if i % 10 == 0:
                self._local_search_phase()
            self.history_best_fitness.append(self.overall_best_fitness)

        self.positions_history.append(np.copy(self.nests))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

    def run_by_nfe(self):

        self._initialize_population()
        self.history_best_fitness = []
        self.positions_history = []

        i = 0
        while self.nfe < self.max_nfe:
            self.positions_history.append(np.copy(self.nests))

            self._levy_flights_phase()
            if self.nfe >= self.max_nfe:
                break

            self._mixing_phase()
            if self.nfe >= self.max_nfe:
                break

            self._discovery_phase()
            if self.nfe >= self.max_nfe:
                break

            if i % 10 == 0:
                self._local_search_phase()
                if self.nfe >= self.max_nfe:
                    break
            self.history_best_fitness.append(self.overall_best_fitness)

            i += 1

            if i > self.max_iterations * 2:
                break

        self.positions_history.append(np.copy(self.nests))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness
