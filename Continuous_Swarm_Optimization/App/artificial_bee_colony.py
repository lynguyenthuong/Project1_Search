"""
    Artificial Bee Colony Optimization
"""
import numpy as np
from opt_visualizer import OptimizationVisualizer
from fitness_function import *

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
                break

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

def run_and_compare():
    """Chạy thuật toán trên các bài toán"""
    max_iters = 50
    max_nfe_limit = 5000 # Giới hạn NFE để chạy thử
    population_size = 40

    func = [ackley_function, rastrigin_function, rosenbrock_function, sphere_function]
    for selected_func in func:
        l_bound, u_bound = get_function_bounds(selected_func)
        print(f"\n{'-'*40}")
        print(f"FUNCTION: {selected_func.__name__.replace('_function', '').title()}")
        print(f"Bounds: [{l_bound}, {u_bound}]")
        print(f"{'-'*40}")

        # --- Chế độ 1: Chạy theo Iteration ---
        print(f"--- Running by Iteration (Limit: {max_iters} iters) ---")
        solver_iter = ArtificialBeeColony(
            fitness_func=selected_func,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=max_iters, # Đặt giới hạn iteration
            max_nfe=10**9, # Đặt NFE rất lớn để không bị dừng bởi NFE
            num_employed=population_size // 2,
            num_onlooker=population_size // 2
        )
        # Sử dụng run_by_iteration
        history_iter, best_sol_iter, best_fit_iter, H_fit_iter = solver_iter.run_by_iteration()

        print(f"| {'Algorithm':<15} | {'Best Fitness':<15} | {'Total NFE Used':<15} | {'Best Solution (X)':<25}")
        print(f"| {'-'*15} | {'-'*15} | {'-'*15} | {'-'*25}")
        print(f"| {'ABC (Iter)':<15} | {best_fit_iter:<15.6f} | {solver_iter.nfe:<15} | [{best_sol_iter[0]:.4f}, {best_sol_iter[1]:.4f}]")

        # abc_visualizer_iter = OptimizationVisualizer(solver_iter)
        # abc_visualizer_iter.animate_optimization_2d(history_iter, title_suffix=" - By Iteration")

        
        # --- Chế độ 2: Chạy theo NFE ---
        print(f"\n--- Running by NFE (Limit: {max_nfe_limit} NFE) ---")
        solver_nfe = ArtificialBeeColony(
            fitness_func=selected_func,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=1000, # Đặt iteration lớn để NFE chạy hết
            max_nfe=max_nfe_limit, # Đặt giới hạn NFE
            num_employed=population_size // 2,
            num_onlooker=population_size // 2
        )
        # Sử dụng run_by_nfe
        history_nfe, best_sol_nfe, best_fit_nfe, H_fit_nfe = solver_nfe.run_by_nfe()        
        print(f"| {'Algorithm':<15} | {'Best Fitness':<15} | {'Total NFE Used':<15} | {'Best Solution (X)':<25}")
        print(f"| {'-'*15} | {'-'*15} | {'-'*15} | {'-'*25}")
        print(f"| {'ABC (NFE)':<15} | {best_fit_nfe:<15.6f} | {solver_nfe.nfe:<15} | [{best_sol_nfe[0]:.4f}, {best_sol_nfe[1]:.4f}]")
        
        # abc_visualizer_nfe = OptimizationVisualizer(solver_nfe)
        # abc_visualizer_nfe.animate_optimization_2d(history_nfe, title_suffix=" - By NFE")
        print(np.array(history_nfe).shape)

if __name__ == "__main__":
    run_and_compare()