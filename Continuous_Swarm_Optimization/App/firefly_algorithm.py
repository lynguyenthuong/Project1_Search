import numpy as np

class FireflyAlgorithm:

    def __init__(self, fitness_func, lower_bound, upper_bound,
                 n_fireflies=50, max_iterations=100, max_nfe=10000,
                 alpha=0.2, beta0=1.0, gamma=1.0, d=2):

        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace(
            '_function', '').title()
        self.dim = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_fireflies = n_fireflies
        self.max_iterations = max_iterations
        self.max_nfe = max_nfe
        self.nfe = 0
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.positions = None
        self.fitness = None
        self.best_pos = None
        self.best_val = float('inf')
        self.positions_history = []
        self.history_best_fitness = []

    def _initialize_fireflies(self):

        self.positions = np.random.uniform(self.lower_bound, self.upper_bound,
                                           (self.n_fireflies, self.dim))

        self.fitness = np.zeros(self.n_fireflies)
        for i in range(self.n_fireflies):
            self.fitness[i] = self.fitness_func(self.positions[i])
            self.nfe += 1
        best_idx = np.argmin(self.fitness)
        self.best_pos = self.positions[best_idx].copy()
        self.best_val = self.fitness[best_idx]

    def _update_fireflies(self):

        new_positions = self.positions.copy()

        for i in range(self.n_fireflies):

            for j in range(self.n_fireflies):
                if self.fitness[j] < self.fitness[i]:
                    r_sq = np.sum((self.positions[i] - self.positions[j])**2)
                    beta = self.beta0 * np.exp(-self.gamma * r_sq)

                    random_step = self.alpha * (np.random.rand(self.dim) - 0.5)
                    new_positions[i] += beta * \
                        (self.positions[j] - self.positions[i]) + random_step

            new_positions[i] = np.clip(
                new_positions[i], self.lower_bound, self.upper_bound)

        self.positions = new_positions

        self.fitness = np.zeros(self.n_fireflies)
        for i in range(self.n_fireflies):
            if self.nfe >= self.max_nfe and self.max_nfe > 0:
                self.fitness[i] = float('inf')
                continue

            self.fitness[i] = self.fitness_func(self.positions[i])
            self.nfe += 1

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_val:
            self.best_val = self.fitness[best_idx]
            self.best_pos = self.positions[best_idx].copy()

    def run_by_iteration(self):

        self._initialize_fireflies()
        self.history_best_fitness = []
        self.positions_history = []

        for _ in range(self.max_iterations):
            self.positions_history.append(np.copy(self.positions))
            self._update_fireflies()
            self.history_best_fitness.append(self.best_val)

        self.positions_history.append(np.copy(self.positions))
        return self.positions_history, self.best_pos, self.best_val, self.history_best_fitness

    def run_by_nfe(self):

        self._initialize_fireflies()
        self.history_best_fitness = []
        self.positions_history = []

        for _ in range(self.max_iterations):
            if self.nfe >= self.max_nfe:
                break

            self.positions_history.append(np.copy(self.positions))
            self._update_fireflies()
            self.history_best_fitness.append(self.best_val)

        self.positions_history.append(np.copy(self.positions))
        return self.positions_history, self.best_pos, self.best_val, self.history_best_fitness
