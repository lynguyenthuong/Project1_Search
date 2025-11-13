import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""## Các hàm cần tối ưu"""

def rosenbrock_function(x):
    """Hàm Rosenbrock

    Tham số:
        x (List hoặc np.array): Điểm (có thể là bất kỳ chiều nào)

    Trả về:
        float hoặc np.array: Giá trị của hàm Rosenbrock
    """
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2, axis=0)


def sphere_function(x):
    """Hàm Sphere

    Tham số:
        x (List hoặc np.array): Điểm

    Trả về:
        float hoặc np.array: Giá trị của hàm Sphere
    """
    x = np.array(x)
    return np.sum(x**2, axis=0)


def rastrigin_function(x):
    """Hàm Rastrigin

    Tham số:
        x (List hoặc np.array): Điểm

    Trả về:
        float hoặc np.array: Giá trị của hàm Rastrigin
    """
    x = np.array(x)
    d = len(x)
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=0)


def ackley_function(x):
    """Hàm Ackley

    Tham số:
        x (List hoặc np.array): Điểm

    Trả về:
        float hoặc np.array: Giá trị của hàm Ackley
    """
    x = np.array(x)
    d = len(x)

    # Các hằng số
    a = 20
    b = 0.2
    c = 2 * np.pi

    sum1 = np.sum(x**2, axis=0)
    sum2 = np.sum(np.cos(c * x), axis=0)

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    return term1 + term2 + a + np.exp(1)


def get_function_bounds(func):
    """Trả về bound thích hợp cho các hàm fitness."""
    func_name = func.__name__

    if func_name == 'ackley_function':
        return -32.768, 32.768
    elif func_name == 'rosenbrock_function':
        return -2.048, 2.048
    elif func_name == 'rastrigin_function':
        return -5.12, 5.12
    elif func_name == 'sphere_function':
        return -5.12, 5.12
    else:
        # Default bounds
        return -10, 10


def get_global_min_pos(func, d=2):
    """Vị trí của global minimum"""
    func_name = func.__name__

    if func_name == 'rosenbrock_function':
        return tuple([1] * d)
    elif func_name in ['ackley_function', 'rastrigin_function', 'sphere_function']:
        return tuple([0] * d)
    else:
        return None

def plot_3d_surface(func, lower_bound, upper_bound):
    """Vẽ đồ thị 3D."""
    x = np.linspace(lower_bound, upper_bound, 500)
    y = np.linspace(lower_bound, upper_bound, 500)
    x_meshgrid, y_meshgrid = np.meshgrid(x, y)
    z = func([x_meshgrid, y_meshgrid])
    func_name = func.__name__.replace('_function', '').title()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_meshgrid, y_meshgrid, z,
                    cmap='viridis', edgecolor='none', alpha=0.8)

    global_min_pos = get_global_min_pos(func)
    global_min_val = func(global_min_pos)
    ax.scatter(global_min_pos[0], global_min_pos[1], global_min_val,
                color='red', marker='*', s=200, label='Global Minimum', zorder=5)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel(f'Fitness Value $f(x)$ ({func_name})')
    ax.set_title(f"Đồ thị {func_name} Function")
    ax.legend()
    plt.tight_layout()
    plt.show()

list_func = [ackley_function, rastrigin_function, rosenbrock_function, sphere_function]
for func in list_func:
    lower_bound, upper_bound = get_function_bounds(func)
    plot_3d_surface(func, lower_bound, upper_bound)

"""## Thuật toán Artificial Bee Colony"""

class ArtificialBeeColony:
    """
    Cài đặt thuật toán Artificial Bee Colony
    """

    def __init__(self, fitness_func, lower_bound, upper_bound, num_employed=50, num_onlooker=50, limit=10, max_iterations=100, max_nfe=10000, d=2):

        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace('_function', '').title()
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
        """Khởi tạo vị trí các nguồn thức ăn và tính giá trị fitness ban đầu."""
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
        self.overall_best_solution = self.food_sources[np.argmin(self.fitness_values)]

    def _update_food_source(self, current_index, k_index):
        """Cập nhật nguồn thức ăn."""
        j = np.random.randint(0, self.dim)
        mutant = np.copy(self.food_sources[current_index])

        phi = np.random.uniform(-1, 1)
        # vij = xij + ϕij (xij − xkj ) : biểu thức 2 trong báo cáo
        mutant[j] = self.food_sources[current_index][j] + phi * (self.food_sources[current_index][j] - self.food_sources[k_index][j])

        # Boundary check
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

        mutant_fitness = self.fitness_func(mutant)
        self.nfe += 1 # Tăng NFE

        # Greedy selection
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
        """Giai đoạn ong thợ của thuật toán ABC."""
        for i in range(self.num_employed_bees):
            k = i
            # Chọn ngẫu nhiên một con ong khác
            while k == i:
                k = np.random.randint(0, self.num_employed_bees)
            self._update_food_source(i, k)

    def _onlooker_bees_phase(self):
        """Giai đoạn ong quan sát của thuật toán ABC."""
        # (quality = 1 / (1 + fitness)) : vì bài toán tìm giá trị tối ưu để minimize fitness function
        # -> fitness càng nhỏ thì xác suất chọn càng cao -> Nghịch đảo fitness
        qualities = 1.0 / (1.0 + self.fitness_values)
        total_quality = np.sum(qualities)

        if total_quality == 0:
            probabilities = np.ones(self.num_employed_bees) / self.num_employed_bees
        else:
            probabilities = qualities / total_quality

        for _ in range(self.num_onlooker_bees):
            selected_index = np.random.choice(self.num_employed_bees, p=probabilities)
            k = selected_index
            # Chọn ngẫu nhiên một con ong thợ khác
            while k == selected_index:
                k = np.random.randint(0, self.num_employed_bees)
            self._update_food_source(selected_index, k)

    def _scout_bees_phase(self):
        """Giai đoạn ong do thám của thuật toán ABC."""
        for k_scout in range(self.num_employed_bees):
            # Rời bỏ nguồn thức ăn hiện tại và trở thành ong do thám
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
                break # break để giới hạn chỉ tạo 1 ong do thám

    def run_by_iteration(self):
        """Vòng lặp chính của thuật toán ABC, dừng theo max_iterations."""
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
        """Vòng lặp chính của thuật toán ABC, dừng theo max_nfe."""
        self._initialize_population()
        self.history_best_fitness = []
        self.positions_history = []
        while self.nfe < self.max_nfe:
            self.positions_history.append(np.copy(self.food_sources))
            self._employed_bees_phase()
            if self.nfe >= self.max_nfe: break
            self._onlooker_bees_phase()
            if self.nfe >= self.max_nfe: break
            self._scout_bees_phase()
            self.history_best_fitness.append(self.overall_best_fitness)
        self.positions_history.append(np.copy(self.food_sources))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

"""## Các thuật toán truyền thống"""

class HillClimbing:
    def __init__(self, fitness_func, lower_bound, upper_bound, step_size=0.1, max_iterations=100, max_nfe=10000, d=2):
        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace('_function', '').title()
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
        perturbation = np.random.uniform(-self.step_size, self.step_size, size=self.dim)
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
        self.function_name = fitness_func.__name__.replace('_function', '').title()
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
        perturbation = np.random.uniform(-self.step_size, self.step_size, size=self.dim)
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
    """
    Cài đặt Genetic Algorithm
    """

    def __init__(self, fitness_func, lower_bound, upper_bound, population_size=100, max_iterations=100, max_nfe=10000, crossover_rate=0.8, mutation_rate=0.1, d=2):
        """
        Khởi tạo Genetic Algorithm
        """
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
        self.function_name = fitness_func.__name__.replace('_function', '').title()

        self.population = None
        self.fitness_values = None
        self.positions_history = []
        self.overall_best_solution = None
        self.overall_best_fitness = float('inf')
        self.history_best_fitness = []

    def _initialize_population(self):
        """Khởi tạo quần thể ngẫu nhiên"""
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
        """Chọn lọc cha mẹ bằng Tournament Selection"""
        tournament_size = 3
        selected_parents = []
        for _ in range(self.pop_size):
            # Chọn ngẫu nhiên các cá thể tham gia tournament
            competitor_indices = np.random.choice(
                self.pop_size,
                size=tournament_size,
                replace=False
            )
            # Tìm cá thể tốt nhất trong tournament
            best_competitor_index = competitor_indices[
                np.argmin(self.fitness_values[competitor_indices])
            ]
            selected_parents.append(self.population[best_competitor_index])
        return np.array(selected_parents)

    def _crossover(self, parent1, parent2):
        """Lai ghép hai cha mẹ bằng BLX-alpha crossover"""
        if np.random.random() < self.crossover_rate:
            alpha = 0.5
            child1 = np.empty(self.dim)
            child2 = np.empty(self.dim)
            for d in range(self.dim):
                # Tính khoảng mở rộng I
                d_val = abs(parent1[d] - parent2[d])
                min_val = min(parent1[d], parent2[d]) - alpha * d_val
                max_val = max(parent1[d], parent2[d]) + alpha * d_val

                # Lấy mẫu từ phân phối đều trong khoảng I
                child1[d] = np.random.uniform(min_val, max_val)
                child2[d] = np.random.uniform(min_val, max_val)
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def _mutation(self, individual):
        """Đột biến cá thể bằng Gaussian perturbation"""
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
        """Đánh giá quần thể và đếm NFE"""
        self.fitness_values = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            # Kiểm tra NFE trước khi đánh giá
            if self.nfe >= self.max_nfe and self.max_nfe > 0:
                # Nếu vượt quá, điền giá trị tệ nhất (inf) và không đếm
                self.fitness_values[i] = float('inf')
            else:
                self.fitness_values[i] = self.fitness_func(self.population[i])
                self.nfe += 1

        # Cập nhật Elitism
        current_best_fitness = np.min(self.fitness_values)
        if current_best_fitness < self.overall_best_fitness:
            self.overall_best_fitness = current_best_fitness
            self.overall_best_solution = self.population[
                np.argmin(self.fitness_values)
            ]

    def _run_one_generation(self):
        """Chạy một thế hệ GA (Selection, Crossover, Mutation, Evaluation)"""
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

        # Đánh giá và Elitism
        self._evaluate_population()

    def run_by_iteration(self):
        """Chạy thuật toán Genetic Algorithm (theo iteration)"""
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
        """Chạy thuật toán Genetic Algorithm (theo NFE)"""
        self._initialize_population()
        self.history_best_fitness = []
        self.positions_history = []

        # Dùng max_iterations như một rào cản an toàn
        for _ in range(self.max_iterations):
            if self.nfe >= self.max_nfe:
                break

            self.positions_history.append(np.copy(self.population))
            self._run_one_generation() # NFE += pop_size
            self.history_best_fitness.append(self.overall_best_fitness)

        self.positions_history.append(np.copy(self.population))
        return  self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

"""## class dùng cho việc tạo GIF quá trình hội tụ của thuật toán"""

class OptimizationVisualizer:
    def __init__(self, instance=None, fitness_func=None, lower_bound=None, upper_bound=None):
        self.instance = instance
        self.func = instance.fitness_func
        self.name = instance.function_name
        self.max_iter = instance.max_iterations
        self.lower_bound = instance.lower_bound
        self.upper_bound = instance.upper_bound

    def animate_optimization_2d(self, positions_history):
        """Tạo GIF quá trình hội tụ"""
        if self.instance:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(self.lower_bound, self.upper_bound)
            ax.set_ylim(self.lower_bound, self.upper_bound)
            ax.set_title(f'Tìm kiếm trên {self.name}')
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')

            x = np.linspace(self.lower_bound, self.upper_bound, 100)
            y = np.linspace(self.lower_bound, self.upper_bound, 100)
            x_meshgrid, y_meshgrid = np.meshgrid(x, y)
            z = self.func([x_meshgrid, y_meshgrid])

            contour = ax.contourf(x_meshgrid, y_meshgrid, z,
                                levels=50, cmap='viridis', alpha=0.8)
            fig.colorbar(contour, label='Fitness Value $f(x)$')

            global_min_pos = get_global_min_pos(self.func)
            ax.plot(global_min_pos[0], global_min_pos[1], 'w*', markersize=12,
                    markeredgecolor='black', label=f'Global Minimum {global_min_pos}')

            scat = ax.scatter([], [], c='yellow', s=50,
                            label='estimator', edgecolor='black')
            best_point, = ax.plot([], [], 'o', color='red', markersize=10,
                                markeredgecolor='black', label='Solution tốt nhất tìm thấy')

            ax.legend(loc='upper right')

            def update(frame):
                current_positions = positions_history[frame]
                scat.set_offsets(current_positions)

                current_fitness = np.array([self.func(p)
                                            for p in current_positions])
                current_best_pos = current_positions[np.argmin(current_fitness)]

                ax.set_title(
                    f'Tìm kiếm trên {self.name}: Vòng lặp {frame + 1}/{self.max_iter}')
                best_point.set_data(
                    np.array([current_best_pos[0]]), np.array([current_best_pos[1]]))

                return scat, best_point

            anim = animation.FuncAnimation(
                fig, update, frames=len(positions_history), interval=150, repeat=False, blit=False
            )

            gif_filename = f'optimization_{self.name.lower()}_2d.gif'

            anim.save(gif_filename, writer='pillow', fps=10)
            plt.show()

def run_all(max_iters = 50, max_nfe_limit = 5000, population_size = 40):
    """Chạy ABC thuật toán trên các bài toán"""

    func = [ackley_function, rastrigin_function, rosenbrock_function, sphere_function]
    for selected_func in func:
        l_bound, u_bound = get_function_bounds(selected_func)
        print(f"\n{'-'*40}")
        print(f"Function: {selected_func.__name__.title()}")
        print(f"Bounds: [{l_bound}, {u_bound}]")
        print(f"{'-'*40}")

        # --- Chế độ 1: Chạy theo Iteration ---
        print(f"--- Chạy theo Iteration (Limit: {max_iters} iters) ---")
        solver_iter = ArtificialBeeColony(
            fitness_func=selected_func,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=max_iters, # Đặt giới hạn iteration
            max_nfe=10**9, # Đặt NFE rất lớn để không bị dừng bởi NFE
            num_employed=population_size // 2,
            num_onlooker=population_size // 2
        )
        history_iter, best_sol_iter, best_fit_iter, H_fit_iter = solver_iter.run_by_iteration()

        print(f"| {'Algorithm':<15} | {'Best Fitness':<15} | {'NFE':<15} | {'Best Solution (X)':<25}")
        print(f"| {'-'*15} | {'-'*15} | {'-'*15} | {'-'*25}")
        print(f"| {'ABC (Iter)':<15} | {best_fit_iter:<15.6f} | {solver_iter.nfe:<15} | [{best_sol_iter[0]:.4f}, {best_sol_iter[1]:.4f}]")

        abc_visualizer_iter = OptimizationVisualizer(solver_iter)
        abc_visualizer_iter.animate_optimization_2d(history_iter)


        # --- Chế độ 2: Chạy theo NFE ---
        print(f"\n--- Chạy theo NFE (Limit: {max_nfe_limit} NFE) ---")
        solver_nfe = ArtificialBeeColony(
            fitness_func=selected_func,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=1000, # Đặt iteration lớn để NFE chạy hết
            max_nfe=max_nfe_limit, # Đặt giới hạn NFE
            num_employed=population_size // 2,
            num_onlooker=population_size // 2
        )
        history_nfe, best_sol_nfe, best_fit_nfe, H_fit_nfe = solver_nfe.run_by_nfe()
        print(f"| {'Algorithm':<15} | {'Best Fitness':<15} | {'NFE':<15} | {'Best Solution (X)':<25}")
        print(f"| {'-'*15} | {'-'*15} | {'-'*15} | {'-'*25}")
        print(f"| {'ABC (NFE)':<15} | {best_fit_nfe:<15.6f} | {solver_nfe.nfe:<15} | [{best_sol_nfe[0]:.4f}, {best_sol_nfe[1]:.4f}]")

        abc_visualizer_nfe = OptimizationVisualizer(solver_nfe)
        abc_visualizer_nfe.animate_optimization_2d(history_nfe)
run_all()
## có file được xuất ra

"""## So sánh ABC với các thuật toán khác"""

def plot_performance_dashboard(results_iter, results_nfe, fitness_func, max_nfe_limit):
    def get_gen_fitness(history_pos, func):
        gen_fitness_history = []
        for population in history_pos:
            pop_arr = np.array(population)
            if pop_arr.ndim == 2 and pop_arr.shape[0] > 1:
                fits = [func(ind) for ind in pop_arr]
                gen_fitness_history.append(np.min(fits))
            else:
                individual = pop_arr[0] if pop_arr.ndim == 2 else pop_arr
                gen_fitness_history.append(func(individual))
        return gen_fitness_history

    # --- ĐỒ THỊ 1: Best So Far (theo Iteration) ---
    plt.figure(figsize=(10, 6))
    for name, (pos_hist, best_hist) in results_iter.items():
        plt.plot(best_hist, label=name)

    plt.title(f"Best-So-Far (theo Iteration) - {fitness_func.__name__}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- ĐỒ THỊ 2: Best of Generation (theo Iteration) ---
    plt.figure(figsize=(10, 6))
    for name, (pos_hist, best_hist) in results_iter.items():
        gen_hist = get_gen_fitness(pos_hist, fitness_func)
        plt.plot(gen_hist, label=name)

    plt.title(f"Best-of-Generation (theo Iteration) - {fitness_func.__name__}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- ĐỒ THỊ 3: Best So Far (theo NFE) ---
    plt.figure(figsize=(10, 6))
    for name, (pos_hist, best_hist) in results_nfe.items():
        x_axis = np.linspace(0, max_nfe_limit, len(best_hist))
        plt.plot(x_axis, best_hist, label=name)

    plt.title(f"Best-So-Far (theo NFE) - {fitness_func.__name__}")
    plt.xlabel("NFE")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- ĐỒ THỊ 4: Best of Generation (theo NFE) ---
    plt.figure(figsize=(10, 6))
    for name, (pos_hist, best_hist) in results_nfe.items():
        gen_hist = get_gen_fitness(pos_hist, fitness_func)
        x_axis = np.linspace(0, max_nfe_limit, len(gen_hist))
        plt.plot(x_axis, gen_hist, label=name)

    plt.title(f"Best-of-Generation (theo NFE) - {fitness_func.__name__}")
    plt.xlabel("NFE")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

target_func = rastrigin_function
lb, ub = get_function_bounds(target_func)
D = 2
MAX_ITER = 50
MAX_NFE = 1000

algorithms = [
    ('ABC', ArtificialBeeColony, {'num_employed': 20, 'num_onlooker': 20, 'limit': 5}),
    ('GA', GeneticAlgorithm, {'population_size': 40}),
    ('HC', HillClimbing, {'step_size': 1}),
    ('SA', SimulatedAnnealing, {'initial_temp': 1000})
]

results_iter_data = {}
results_nfe_data = {}

for name, AlgoClass, params in algorithms:
    solver = AlgoClass(target_func, lb, ub, d=D, max_iterations=MAX_ITER, **params)
    pos_hist, _, _, best_hist = solver.run_by_iteration()
    results_iter_data[name] = (pos_hist, best_hist)

for name, AlgoClass, params in algorithms:
    solver = AlgoClass(target_func, lb, ub, d=D, max_nfe=MAX_NFE, **params)
    pos_hist, _, _, best_hist = solver.run_by_nfe()
    results_nfe_data[name] = (pos_hist, best_hist)

plot_performance_dashboard(
    results_iter=results_iter_data,
    results_nfe=results_nfe_data,
    fitness_func=target_func,
    max_nfe_limit=MAX_NFE
)

class SensitivityAnalyzer:
    def __init__(self, algorithm_class, fixed_params, base_seed=42):
        self.algorithm_class = algorithm_class
        self.fixed_params = fixed_params
        self.base_seed = base_seed

    def _prepare_params(self, param_name, val, current_params):
        params = current_params.copy()

        if param_name == 'population_size' and 'ArtificialBeeColony' in self.algorithm_class.__name__:
            if 'num_employed' in params: del params['num_employed']
            if 'num_onlooker' in params: del params['num_onlooker']
            params['num_employed'] = int(val / 2)
            params['num_onlooker'] = int(val / 2)
        else:
            params[param_name] = val

        return params

    def _run_trials(self, fitness_func, params, repeats):
        fits = []
        for r in range(repeats):
            np.random.seed(self.base_seed + r)
            lb, ub = get_function_bounds(fitness_func)
            run_params = params.copy()
            if 'lower_bound' not in self.fixed_params: run_params['lower_bound'] = lb
            if 'upper_bound' not in self.fixed_params: run_params['upper_bound'] = ub

            solver = self.algorithm_class(fitness_func=fitness_func, **run_params)
            _, _, best_fit, _ = solver.run_by_iteration()
            fits.append(best_fit)
        return np.mean(fits)

    def analyze_single_param(self, fitness_func, param_name, param_values, repeats=5):
        print(f"\n--- Phân tích đơn: '{param_name}' trên hàm {fitness_func.__name__} ---")
        results = []

        for val in param_values:
            print(f" > Testing {param_name} = {val}...", end='\r')
            params = self._prepare_params(param_name, val, self.fixed_params)
            mean_fit = self._run_trials(fitness_func, params, repeats)
            results.append(mean_fit)

        plt.figure(figsize=(8, 5))
        plt.plot(param_values, results, marker='o', linestyle='-', color='b')
        plt.xlabel(param_name)
        plt.ylabel(f"Mean Best Fitness ({repeats} runs)")
        plt.title(f"Độ nhạy: {param_name} ({self.algorithm_class.__name__})")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def analyze_dual_params(self, fitness_func, param1, param2, repeats=5):
        p1_name, p1_vals = param1
        p2_name, p2_vals = param2

        print(f"\n--- Phân tích kép: '{p1_name}' vs '{p2_name}' ---")
        matrix = np.zeros((len(p1_vals), len(p2_vals)))

        for i, val1 in enumerate(p1_vals):
            for j, val2 in enumerate(p2_vals):
                print(f" > {p1_name}={val1}, {p2_name}={val2}...", end='\r')

                params = self.fixed_params.copy()
                params = self._prepare_params(p1_name, val1, params)
                params = self._prepare_params(p2_name, val2, params)

                matrix[i, j] = self._run_trials(fitness_func, params, repeats)

        plt.figure(figsize=(10, 7))
        im = plt.imshow(matrix, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(im, label=f'Mean Best Fitness')

        plt.xticks(np.arange(len(p2_vals)), p2_vals)
        plt.yticks(np.arange(len(p1_vals)), p1_vals)

        plt.xlabel(p2_name, fontweight='bold')
        plt.ylabel(p1_name, fontweight='bold')
        plt.title(f"Tương quan: {p1_name} vs {p2_name}")

        for i in range(len(p1_vals)):
            for j in range(len(p2_vals)):
                c = "white" if matrix[i, j] < np.mean(matrix) else "black"
                plt.text(j, i, f"{matrix[i, j]:.2e}", ha="center", va="center", color=c)

        plt.tight_layout()
        plt.show()

    def compare_functions(self, fitness_funcs, param_name, param_values, repeats=5):
        print(f"\n--- So sánh đa hàm mục tiêu trên tham số '{param_name}' ---")
        plt.figure(figsize=(10, 6))

        param_values = sorted(param_values)

        for func in fitness_funcs:
            func_name = func.__name__.replace('_function', '').title()
            print(f" > Processing function: {func_name}...")
            mean_results = []

            for val in param_values:
                params = self._prepare_params(param_name, val, self.fixed_params)
                mean_fit = self._run_trials(func, params, repeats)
                mean_results.append(mean_fit)

            plt.plot(param_values, mean_results, marker='o', linewidth=2, markersize=6, label=func_name)

        plt.title(f"Độ nhạy của tham số '{param_name.upper()}' trên các hàm khác nhau")
        plt.xlabel(param_name)
        plt.ylabel("Mean Best Fitness")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
base_config = {
    'd': 2,
    'max_iterations': 100,
    # Lưu ý: num_employed/onlooker có thể bị ghi đè nếu test population_size
    'num_employed': 25,
    'num_onlooker': 25,
    'limit': 10 # Giá trị mặc định nếu không test limit
}

analyzer = SensitivityAnalyzer(
    algorithm_class=ArtificialBeeColony,
    fixed_params=base_config,
    base_seed=42
)

analyzer.analyze_single_param(
    fitness_func=rastrigin_function,
    param_name='limit',
    param_values=np.arange(1,10,1),
    repeats=10
)

analyzer.compare_functions(
    fitness_funcs=[ackley_function, rastrigin_function, rosenbrock_function, sphere_function],
    param_name='limit',
    param_values=np.arange(1,10,1),
    repeats=10
)

analyzer.analyze_dual_params(
    fitness_func=rastrigin_function,
    param1=('num_employed', [10, 20, 30, 40, 50]),
    param2=('num_onlooker', [10, 20, 30, 40, 50]),
    repeats=10
)

analyzer.compare_functions(
    fitness_funcs=[ackley_function, rastrigin_function, rosenbrock_function, sphere_function],
    param_name='population_size',
    param_values=np.arange(10,100,10),
    repeats=10
)