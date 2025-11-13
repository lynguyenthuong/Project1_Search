import numpy as np
from fitness_function import *
from opt_visualizer import OptimizationVisualizer

class ParticleSwarmOptimization:
    def __init__(self, fitness_func, lower_bound, upper_bound,
                 n_particles=50, max_iterations=100, max_nfe=10000,
                 w=0.7, c1=1.5, c2=1.5, d=2):

        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace('_function', '').title()
        self.dim = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.max_nfe = max_nfe
        self.nfe = 0
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.positions = None
        self.velocities = None
        self.pbest_pos = None
        self.pbest_val = None
        self.gbest_pos = None
        self.gbest_val = float('inf')
        self.positions_history = []
        self.history_best_fitness = [] 

    def _initialize_swarm(self):
        """Khởi tạo quần thể hạt"""
        self.positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.n_particles, self.dim))
        self.velocities = np.zeros_like(self.positions)
        self.pbest_pos = np.copy(self.positions)
        
        self.pbest_val = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            self.pbest_val[i] = self.fitness_func(self.positions[i])
            self.nfe += 1 # Đếm NFE

        self.gbest_pos = self.positions[np.argmin(self.pbest_val)].copy()
        self.gbest_val = np.min(self.pbest_val)

    def _update_particles(self):
        """Cập nhật vận tốc và vị trí các hạt"""
        r1, r2 = np.random.rand(), np.random.rand()
        self.velocities = (
            self.w * self.velocities
            + self.c1 * r1 * (self.pbest_pos - self.positions)
            + self.c2 * r2 * (self.gbest_pos - self.positions)
        )
        self.positions += self.velocities
        self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        # Đánh giá các vị trí mới
        current_vals = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            if self.nfe >= self.max_nfe and self.max_nfe > 0:
                current_vals[i] = float('inf')
                continue
            current_vals[i] = self.fitness_func(self.positions[i])
            self.nfe += 1 # Đếm NFE

        # Cập nhật pbest
        better = current_vals < self.pbest_val
        self.pbest_pos[better] = self.positions[better]
        self.pbest_val[better] = current_vals[better]

        # Cập nhật gbest
        current_best_val = np.min(current_vals)
        if current_best_val < self.gbest_val:
            self.gbest_val = current_best_val
            self.gbest_pos = self.positions[np.argmin(current_vals)].copy()

    def run_by_iteration(self):
        """Vòng lặp chính của thuật toán PSO (theo iteration)"""
        self._initialize_swarm()
        self.history_best_fitness = [] 
        self.positions_history = []

        for _ in range(self.max_iterations):
            self.positions_history.append(np.copy(self.positions))
            self._update_particles()
            self.history_best_fitness.append(self.gbest_val)

        self.positions_history.append(np.copy(self.positions))
        return self.positions_history, self.gbest_pos, self.gbest_val, self.history_best_fitness

    def run_by_nfe(self):
        """Vòng lặp chính của thuật toán PSO (theo NFE)"""
        self._initialize_swarm() # NFE = n_particles
        self.history_best_fitness = [] 
        self.positions_history = []
        
        # Dùng max_iterations làm rào cản an toàn
        for _ in range(self.max_iterations):
            if self.nfe >= self.max_nfe:
                break
                
            self.positions_history.append(np.copy(self.positions))
            self._update_particles() # NFE += n_particles
            self.history_best_fitness.append(self.gbest_val) 

        self.positions_history.append(np.copy(self.positions))
        return self.positions_history, self.gbest_pos, self.gbest_val, self.history_best_fitness
    
def run_and_compare():
    """Chạy thử PSO trên các hàm fitness và hiển thị kết quả"""
    max_iters = 50
    max_nfe_limit = 5000
    population_size = 40
    
    func_list = [ackley_function, rastrigin_function, rosenbrock_function, sphere_function]

    for f in func_list:
        l_bound, u_bound = get_function_bounds(f)
        print(f"\n{'-'*40}")
        print(f"FUNCTION: {f.__name__.replace('_function', '').title()} (PSO)")
        print(f"Bounds: [{l_bound}, {u_bound}]")
        print(f"{'-'*40}")

        # --- Chế độ 1: Chạy theo Iteration ---
        print(f"--- Running by Iteration (Limit: {max_iters} iters) ---")
        solver_iter = ParticleSwarmOptimization(
            fitness_func=f,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=max_iters,
            max_nfe=9999999, # NFE không giới hạn
            n_particles=population_size
        )
        history_iter, best_pos_iter, best_fit_iter, H_fit_iter = solver_iter.run_by_iteration()
        print(f"| {'Mode':<10} | {'Best Fitness':<15} | {'Total NFE':<10} | {'Best Position':<25}")
        print(f"| {'-'*10} | {'-'*15} | {'-'*10} | {'-'*25}")
        print(f"| {'Iter':<10} | {best_fit_iter:<15.6f} | {solver_iter.nfe:<10} | [{best_pos_iter[0]:.4f}, {best_pos_iter[1]:.4f}]")

        # --- Chế độ 2: Chạy theo NFE ---
        print(f"\n--- Running by NFE (Limit: {max_nfe_limit} NFE) ---")
        solver_nfe = ParticleSwarmOptimization(
            fitness_func=f,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=1000, # Iter không giới hạn
            max_nfe=max_nfe_limit,
            n_particles=population_size
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