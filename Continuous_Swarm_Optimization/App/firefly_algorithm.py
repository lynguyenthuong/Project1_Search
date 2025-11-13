import numpy as np
from fitness_function import *
from opt_visualizer import OptimizationVisualizer


class FireflyAlgorithm:
    """Thuật toán Firefly Algorithm (FA)"""
    def __init__(self, fitness_func, lower_bound, upper_bound,
                 n_fireflies=50, max_iterations=100, max_nfe=10000, 
                 alpha=0.2, beta0=1.0, gamma=1.0, d=2):

        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace('_function', '').title()
        self.dim = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_fireflies = n_fireflies
        self.max_iterations = max_iterations
        self.max_nfe = max_nfe
        self.nfe = 0
        self.alpha = alpha       # hệ số ngẫu nhiên
        self.beta0 = beta0       # độ hấp dẫn ban đầu
        self.gamma = gamma       # hệ số suy giảm độ hấp dẫn
        self.positions = None
        self.fitness = None
        self.best_pos = None
        self.best_val = float('inf')
        self.positions_history = []
        self.history_best_fitness = []

    def _initialize_fireflies(self):
        """Khởi tạo quần thể đom đóm"""
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
        """Cập nhật vị trí các đom đóm"""
        # Tạo bản sao vị trí để cập nhật đồng thời (tránh ảnh hưởng lẫn nhau trong 1 vòng)
        new_positions = self.positions.copy()
        
        for i in range(self.n_fireflies):
            # Di chuyển về phía những con sáng hơn
            for j in range(self.n_fireflies):
                if self.fitness[j] < self.fitness[i]:  # j sáng hơn i
                    r_sq = np.sum((self.positions[i] - self.positions[j])**2)
                    beta = self.beta0 * np.exp(-self.gamma * r_sq)
                    
                    # Cập nhật vị trí đom đóm i
                    random_step = self.alpha * (np.random.rand(self.dim) - 0.5)
                    new_positions[i] += beta * (self.positions[j] - self.positions[i]) + random_step

            # Giữ trong biên
            new_positions[i] = np.clip(new_positions[i], self.lower_bound, self.upper_bound)

        self.positions = new_positions

        # Tính lại fitness cho tất cả
        self.fitness = np.zeros(self.n_fireflies)
        for i in range(self.n_fireflies):
            if self.nfe >= self.max_nfe and self.max_nfe > 0:
                self.fitness[i] = float('inf')
                continue
            
            self.fitness[i] = self.fitness_func(self.positions[i])
            self.nfe += 1

        # Cập nhật best
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_val:
            self.best_val = self.fitness[best_idx]
            self.best_pos = self.positions[best_idx].copy()

    def run_by_iteration(self):
        """Vòng lặp chính của thuật toán FA (theo iteration)"""
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
        """Vòng lặp chính của thuật toán FA (theo NFE)"""
        self._initialize_fireflies() # NFE = n_fireflies
        self.history_best_fitness = []
        self.positions_history = []
        
        # Dùng max_iterations làm rào cản an toàn
        for _ in range(self.max_iterations):
            if self.nfe >= self.max_nfe:
                break
                
            self.positions_history.append(np.copy(self.positions))
            self._update_fireflies() # NFE += n_fireflies
            self.history_best_fitness.append(self.best_val)

        self.positions_history.append(np.copy(self.positions))
        return self.positions_history, self.best_pos, self.best_val, self.history_best_fitness

def run_and_compare():
    """Chạy thử FA trên các hàm fitness và hiển thị kết quả"""
    max_iters = 50
    max_nfe_limit = 5000
    population_size = 40
    
    func_list = [ackley_function, rastrigin_function, rosenbrock_function, sphere_function]

    for f in func_list:
        l_bound, u_bound = get_function_bounds(f)
        print(f"\n{'-'*40}")
        print(f"FUNCTION: {f.__name__.replace('_function', '').title()} (FA)")
        print(f"Bounds: [{l_bound}, {u_bound}]")
        print(f"{'-'*40}")

        # --- Chế độ 1: Chạy theo Iteration ---
        print(f"--- Running by Iteration (Limit: {max_iters} iters) ---")
        solver_iter = FireflyAlgorithm(
            fitness_func=f,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=max_iters,
            max_nfe=9999999, # NFE không giới hạn
            n_fireflies=population_size
        )
        history_iter, best_pos_iter, best_fit_iter, H_fit_iter = solver_iter.run_by_iteration()
        print(f"| {'Mode':<10} | {'Best Fitness':<15} | {'Total NFE':<10} | {'Best Position':<25}")
        print(f"| {'-'*10} | {'-'*15} | {'-'*10} | {'-'*25}")
        print(f"| {'Iter':<10} | {best_fit_iter:<15.6f} | {solver_iter.nfe:<10} | [{best_pos_iter[0]:.4f}, {best_pos_iter[1]:.4f}]")

        # --- Chế độ 2: Chạy theo NFE ---
        print(f"\n--- Running by NFE (Limit: {max_nfe_limit} NFE) ---")
        solver_nfe = FireflyAlgorithm(
            fitness_func=f,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=1000, # Iter không giới hạn
            max_nfe=max_nfe_limit,
            n_fireflies=population_size
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