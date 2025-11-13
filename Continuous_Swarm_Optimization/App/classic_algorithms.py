"""Các thuật toán cổ điển"""
import numpy as np

class HillClimbing:
    """
    Cài đặt thuật toán Hill Climbing (tìm kiếm cục bộ)
    """

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
        """Khởi tạo giải pháp ban đầu ngẫu nhiên."""
        self.current_solution = np.random.uniform(
            low=self.lower_bound, high=self.upper_bound, size=self.dim
        )
        self.current_fitness = self.fitness_func(self.current_solution)
        self.nfe += 1 # Đếm NFE
        self.overall_best_solution = self.current_solution
        self.overall_best_fitness = self.current_fitness

    def _get_neighbor(self):
        """Tạo một giải pháp lân cận bằng cách dịch chuyển ngẫu nhiên một chút."""
        neighbor = np.copy(self.current_solution)
        # Tạo dịch chuyển ngẫu nhiên (hoặc sử dụng Gaussian/phân phối đều)
        # Dịch chuyển ngẫu nhiên theo mỗi chiều
        
        # Tạo nhiễu ngẫu nhiên với biên độ dựa trên step_size
        perturbation = np.random.uniform(-self.step_size, self.step_size, size=self.dim)
        neighbor += perturbation

        # Kiểm tra giới hạn
        neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
        return neighbor

    def run_by_iteration(self):
        """Vòng lặp chính của thuật toán Hill Climbing (theo iteration)"""
        self._initialize()
        self.history_best_fitness = []
        self.positions_history = []

        for _ in range(self.max_iterations):
            self.positions_history.append(np.array([self.current_solution])) # Lưu vị trí
            
            neighbor = self._get_neighbor()
            neighbor_fitness = self.fitness_func(neighbor)
            self.nfe += 1 # Đếm NFE
            # Chọn lựa tham lam (Greedy selection): chỉ chấp nhận lân cận tốt hơn
            if neighbor_fitness < self.current_fitness:
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness
                # Cập nhật giá trị tốt nhất tổng thể
                if self.current_fitness < self.overall_best_fitness:
                    self.overall_best_fitness = self.current_fitness
                    self.overall_best_solution = self.current_solution

            # (Không có bước dịch chuyển nếu không tìm thấy lân cận tốt hơn,
            # thuật toán sẽ dừng lại khi đạt cực tiểu cục bộ nếu không chấp nhận 
            # giải pháp tệ hơn, điều này là đặc trưng của HC)
            self.history_best_fitness.append(self.overall_best_fitness)
            
        self.positions_history.append(np.array([self.current_solution])) # Lưu vị trí cuối cùng
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

    def run_by_nfe(self):
        """Vòng lặp chính của thuật toán Hill Climbing (theo NFE)"""
        self._initialize() # NFE = 1
        self.history_best_fitness = []
        self.positions_history = []

        while self.nfe < self.max_nfe:
            self.positions_history.append(np.array([self.current_solution]))
            
            neighbor = self._get_neighbor()
            neighbor_fitness = self.fitness_func(neighbor)
            self.nfe += 1 # Đếm NFE
            
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
    """
    Cài đặt thuật toán Simulated Annealing
    """

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
        """Khởi tạo giải pháp ban đầu ngẫu nhiên."""
        self.current_solution = np.random.uniform(
            low=self.lower_bound, high=self.upper_bound, size=self.dim
        )
        self.current_fitness = self.fitness_func(self.current_solution)
        self.nfe += 1 # Đếm NFE
        self.overall_best_solution = self.current_solution.copy()
        self.overall_best_fitness = self.current_fitness
    
    def _get_neighbor(self):
        """Tạo một giải pháp lân cận."""
        neighbor = np.copy(self.current_solution)
        # Tạo nhiễu ngẫu nhiên
        perturbation = np.random.uniform(-self.step_size, self.step_size, size=self.dim)
        neighbor += perturbation

        # Kiểm tra giới hạn
        neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
        return neighbor

    def _acceptance_probability(self, new_fitness, iteration):
        """Tính xác suất chấp nhận theo tiêu chí Metropolis."""
        # Epsilon để tránh lỗi chia cho 0, nhưng T không bao giờ là 0
        current_T = self.initial_temp * (self.cooling_rate ** iteration)
        # Nếu giải pháp lân cận tốt hơn, chấp nhận 100%
        if new_fitness < self.current_fitness:
            return 1.0
        
        # Nếu giải pháp lân cận tệ hơn (tăng fitness), chấp nhận với xác suất
        # P = exp(-(new_fitness - self.current_fitness) / T)
        delta_E = new_fitness - self.current_fitness
        
        if current_T > 0:
            return np.exp(-delta_E / current_T)
        else:
            return 0.0 # Nếu T quá nhỏ (gần 0), xác suất chấp nhận giải pháp tệ hơn là 0

    def run_by_iteration(self):
        """Vòng lặp chính của thuật toán Simulated Annealing (theo iteration)"""
        self._initialize()
        self.history_best_fitness = []
        self.positions_history = [] 
        
        for iteration in range(self.max_iterations):
            self.positions_history.append(np.array([self.current_solution])) # Lưu vị trí
            
            neighbor = self._get_neighbor()
            neighbor_fitness = self.fitness_func(neighbor)
            self.nfe += 1 # Đếm NFE

            prob = self._acceptance_probability(neighbor_fitness, iteration)
            # Quyết định chấp nhận giải pháp lân cận
            if np.random.random() < prob:
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness

                # Cập nhật giá trị tốt nhất tổng thể                
                if self.current_fitness < self.overall_best_fitness:
                    self.overall_best_fitness = self.current_fitness
                    self.overall_best_solution = self.current_solution.copy()
                    
            # Giảm nhiệt độ (đã được tích hợp trong _acceptance_probability)
            self.history_best_fitness.append(self.overall_best_fitness)
            
                
        self.positions_history.append(np.array([self.current_solution]))
        # Lưu vị trí cuối cùng
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

    def run_by_nfe(self):
        """Vòng lặp chính của thuật toán Simulated Annealing (theo NFE)"""
        self._initialize() # NFE = 1
        self.history_best_fitness = [] 
        self.positions_history = []
        
        iteration = 0
        while self.nfe < self.max_nfe:
            self.positions_history.append(np.array([self.current_solution]))
            
            neighbor = self._get_neighbor()
            neighbor_fitness = self.fitness_func(neighbor)
            self.nfe += 1 # Đếm NFE

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
            self.nfe += 1 # Đếm NFE

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
        self._initialize_population() # NFE = pop_size
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