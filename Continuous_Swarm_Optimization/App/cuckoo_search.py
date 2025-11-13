import numpy as np
import math
from opt_visualizer import OptimizationVisualizer
from fitness_function import *

class CuckooSearch:
    """
    Cài đặt thuật toán Cuckoo Search (CS)
    """
    def __init__(self, fitness_func, lower_bound, upper_bound, d=2,
                 n=25, pa=0.15, alpha=0.5, max_iterations=100, max_nfe=10000,
                 F=0.5, CR=0.9, levy_beta=1.5, seed=None):
        """
        Khởi tạo các tham số cho thuật toán Cuckoo Search.
        """
        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace('_function', '').title()
        self.dim = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        self.n = n  # Số lượng tổ (population size)
        self.pa = pa  # Xác suất phát hiện
        self.alpha = alpha  # Hệ số bước nhảy
        self.max_iterations = max_iterations
        self.max_nfe = max_nfe
        self.nfe = 0    
        self.F = F  # Hệ số DE
        self.CR = CR  # Tỷ lệ lai ghép DE
        self.levy_beta = levy_beta
        self.seed = seed
        
        self.nests = None
        self.fitness = None 
        
        self.positions_history = []
        self.overall_best_solution = None
        self.overall_best_fitness = float('inf')
        self.history_best_fitness = []

    def _levy_flight(self):
        """Tính toán bước nhảy Levy Flight (Private method)."""
        Lambda = self.levy_beta
        dim = self.dim
        sigma = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
                 (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / Lambda)
        return step

    def _initialize_population(self):
        """Khởi tạo quần thể tổ và tính fitness ban đầu."""
        if self.seed is not None:
            np.random.seed(self.seed)
            
        self.nests = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.n, self.dim))
        
        self.fitness = np.zeros(self.n)
        for i in range(self.n):
            self.fitness[i] = self.fitness_func(self.nests[i])
            self.nfe += 1 # Đếm NFE

        best_idx = np.argmin(self.fitness)
        self.overall_best_fitness = self.fitness[best_idx]
        self.overall_best_solution = self.nests[best_idx].copy()
        
    def _update_best(self, new_solution, new_fitness):
        """Helper để cập nhật giải pháp tốt nhất toàn cục."""
        if new_fitness < self.overall_best_fitness:
            self.overall_best_fitness = new_fitness
            self.overall_best_solution = new_solution.copy()
            
    def _levy_flights_phase(self):
        """Giai đoạn 1: Tạo cuckoo mới bằng Lévy flights."""
        if self.nfe >= self.max_nfe: return # Kiểm tra NFE

        for i in range(self.n):
            if self.nfe >= self.max_nfe: break
                
            step = self._levy_flight()
            phi = np.random.rand()
            
            # Chọn ngẫu nhiên một tổ khác để tính toán
            random_nest_idx = np.random.choice(self.nests.shape[0])
            
            new_nest = self.nests[i] + self.alpha * step * (self.nests[i] - self.nests[random_nest_idx])
            # Sử dụng thông tin của best (tăng cường tìm kiếm cục bộ)
            new_nest += phi * (self.overall_best_solution - self.nests[i]) * 0.5
            new_nest = np.clip(new_nest, self.lower_bound, self.upper_bound)
            
            new_fit = self.fitness_func(new_nest)
            self.nfe += 1 # Đếm NFE
            
            if new_fit < self.fitness[i]:
                self.nests[i] = new_nest
                self.fitness[i] = new_fit
                self._update_best(new_nest, new_fit)

    def _mixing_phase(self):
        """Giai đoạn 2: Trộn quần thể (Sử dụng Differential Evolution)."""
        if self.nfe >= self.max_nfe: return

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
            if self.nfe >= self.max_nfe: break
            new_fitness[i] = self.fitness_func(new_nests[i])
            self.nfe += 1 # Đếm NFE
        
        improved = new_fitness < self.fitness
        self.fitness[improved] = new_fitness[improved]
        self.nests[improved] = new_nests[improved]

        # Kiểm tra best mới sau khi trộn
        best_idx_after_mix = np.argmin(self.fitness)
        if self.fitness[best_idx_after_mix] < self.overall_best_fitness:
            self._update_best(self.nests[best_idx_after_mix], self.fitness[best_idx_after_mix])

    def _discovery_phase(self):
        """Giai đoạn 3: Phát hiện và loại bỏ tổ (Discovery)."""
        if self.nfe >= self.max_nfe: return
        
        num_replace = max(1, int(self.pa * self.n))
        worst_idx = np.argsort(self.fitness)[-num_replace:]
        
        for idx in worst_idx:
            if self.nfe >= self.max_nfe: break
            
            if np.random.rand() < 0.5:
                # Thay bằng tổ mới ngẫu nhiên
                self.nests[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            else:
                # Thay bằng tổ mới gần best (cải tiến)
                self.nests[idx] = np.clip(self.overall_best_solution + 
                                          np.random.normal(0, 0.1 * (self.upper_bound - self.lower_bound), size=self.dim), 
                                          self.lower_bound, self.upper_bound)
            
            self.fitness[idx] = self.fitness_func(self.nests[idx])
            self.nfe += 1 # Đếm NFE
            # Cập nhật best nếu tổ mới (thay thế tổ tệ nhất) lại tốt nhất
            self._update_best(self.nests[idx], self.fitness[idx])

    def _local_search_phase(self):
        """Cải tiến: Tìm kiếm cục bộ định kỳ."""
        if self.nfe >= self.max_nfe: return
        
        local = self.overall_best_solution + np.random.normal(0, 0.01 * (self.upper_bound - self.lower_bound), size=(5, self.dim))
        local = np.clip(local, self.lower_bound, self.upper_bound)
        
        local_f = np.zeros(5)
        for i in range(5):
            if self.nfe >= self.max_nfe: break
            local_f[i] = self.fitness_func(local[i])
            self.nfe += 1 # Đếm NFE
        
        mi = np.argmin(local_f)
        if local_f[mi] < self.overall_best_fitness:
            self.overall_best_solution = local[mi].copy()
            self.overall_best_fitness = local_f[mi]
            
            worst = np.argmax(self.fitness)
            self.nests[worst] = local[mi].copy()
            self.fitness[worst] = local_f[mi]

    def run_by_iteration(self):
        """Vòng lặp chính của thuật toán CS (theo iteration)."""
        self._initialize_population()
        self.history_best_fitness = []
        self.positions_history = []
        
        for i in range(self.max_iterations):
            self.positions_history.append(np.copy(self.nests))
            
            self._levy_flights_phase()
            self._mixing_phase()
            self._discovery_phase()
            
            # Chạy tìm kiếm cục bộ mỗi 10 vòng lặp
            if i % 10 == 0:
                self._local_search_phase()
            self.history_best_fitness.append(self.overall_best_fitness)
        
        self.positions_history.append(np.copy(self.nests))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

    def run_by_nfe(self):
        """Vòng lặp chính của thuật toán CS (theo NFE)."""
        self._initialize_population() # NFE = n
        self.history_best_fitness = []
        self.positions_history = []
        
        i = 0
        while self.nfe < self.max_nfe:
            self.positions_history.append(np.copy(self.nests))
            
            self._levy_flights_phase()
            if self.nfe >= self.max_nfe: break
                
            self._mixing_phase()
            if self.nfe >= self.max_nfe: break
            
            self._discovery_phase()
            if self.nfe >= self.max_nfe: break
            
            if i % 10 == 0:
                self._local_search_phase()
                if self.nfe >= self.max_nfe: break
            self.history_best_fitness.append(self.overall_best_fitness) 
            
            i += 1
            # Thêm rào cản an toàn
            if i > self.max_iterations * 2: # Gấp đôi max_iterations
                break
        
        self.positions_history.append(np.copy(self.nests))
        return self.positions_history, self.overall_best_solution, self.overall_best_fitness, self.history_best_fitness

def run_and_compare():
    """Chạy thử CS trên các hàm fitness và hiển thị kết quả"""
    max_iters = 50
    max_nfe_limit = 5000
    population_size = 40
    dim = 2

    func = [ackley_function, rastrigin_function, rosenbrock_function, sphere_function]
    
    for selected_func in func:
        l_bound, u_bound = get_function_bounds(selected_func)
        print(f"\n{'-'*40}")
        print(f"FUNCTION: {selected_func.__name__.replace('_function', '').title()} (CS)")
        print(f"Bounds: [{l_bound}, {u_bound}]")
        print(f"{'-'*40}")

        # --- Chế độ 1: Chạy theo Iteration ---
        print(f"--- Running by Iteration (Limit: {max_iters} iters) ---")
        solver_iter = CuckooSearch(
            fitness_func=selected_func,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=max_iters,
            max_nfe=9999999, # NFE không giới hạn
            n=population_size,
            d=dim
        )
        history_iter, best_pos_iter, best_fit_iter, H_fit_iter = solver_iter.run_by_iteration()
        print(f"| {'Mode':<10} | {'Best Fitness':<15} | {'Total NFE':<10} | {'Best Position':<25}")
        print(f"| {'-'*10} | {'-'*15} | {'-'*10} | {'-'*25}")
        print(f"| {'Iter':<10} | {best_fit_iter:<15.6f} | {solver_iter.nfe:<10} | [{best_pos_iter[0]:.4f}, {best_pos_iter[1]:.4f}]")

        # --- Chế độ 2: Chạy theo NFE ---
        print(f"\n--- Running by NFE (Limit: {max_nfe_limit} NFE) ---")
        solver_nfe = CuckooSearch(
            fitness_func=selected_func,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=1000, # Iter không giới hạn
            max_nfe=max_nfe_limit,
            n=population_size,
            d=dim
        )
        history_nfe, best_pos_nfe, best_fit_nfe, H_fit_nfe = solver_nfe.run_by_nfe()
        print(f"| {'Mode':<10} | {'Best Fitness':<15} | {'Total NFE':<10} | {'Best Position':<25}")
        print(f"| {'-'*10} | {'-'*15} | {'-'*10} | {'-'*25}")
        print(f"| {'NFE':<10} | {best_fit_nfe:<15.6f} | {solver_nfe.nfe:<10} | [{best_pos_nfe[0]:.4f}, {best_pos_nfe[1]:.4f}]")

        # vis = OptimizationVisualizer(solver_nfe)
        # vis.animate_optimization_2d(history_nfe)     
        # vis.plot_3d_surface()


if __name__ == "__main__":
    run_and_compare()