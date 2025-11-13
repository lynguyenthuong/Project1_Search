
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fitness_function import *


class ParticleSwarmOptimization:
    def __init__(self, fitness_func, lower_bound, upper_bound,
                 n_particles=50, max_iterations=100, w=0.7, c1=1.5, c2=1.5, d=2):

        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace('_function', '').title()
        self.dim = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w, self.c1, self.c2 = w, c1, c2

        self.positions = None
        self.velocities = None
        self.pbest_pos = None
        self.pbest_val = None
        self.gbest_pos = None
        self.gbest_val = np.inf
        self.positions_history = []

    def _initialize_swarm(self):
        """Khởi tạo quần thể hạt"""
        self.positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.n_particles, self.dim))
        self.velocities = np.zeros_like(self.positions)
        self.pbest_pos = np.copy(self.positions)
        self.pbest_val = np.array([self.fitness_func(p) for p in self.positions])
        self.gbest_pos = self.positions[np.argmin(self.pbest_val)]
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

        current_vals = np.array([self.fitness_func(p) for p in self.positions])
        better = current_vals < self.pbest_val
        self.pbest_pos[better] = self.positions[better]
        self.pbest_val[better] = current_vals[better]

        if np.min(current_vals) < self.gbest_val:
            self.gbest_val = np.min(current_vals)
            self.gbest_pos = self.positions[np.argmin(current_vals)]

    def run(self):
        """Vòng lặp chính của thuật toán PSO"""
        self._initialize_swarm()

        for _ in range(self.max_iterations):
            self.positions_history.append(np.copy(self.positions))
            self._update_particles()

        self.positions_history.append(np.copy(self.positions))
        return self.positions_history, self.gbest_pos, self.gbest_val

class PSOVisualizer:
    """Visualizer cho thuật toán Particle Swarm Optimization (PSO)"""

    def __init__(self, pso_instance):
        self.pso = pso_instance
        self.func = pso_instance.fitness_func
        self.name = pso_instance.function_name
        self.lower_bound = pso_instance.lower_bound
        self.upper_bound = pso_instance.upper_bound
        self.max_iter = pso_instance.max_iterations

    def animate_2d(self):
        """Tạo GIF mô phỏng quá trình hội tụ của PSO trên không gian 2D (tự động điều chỉnh tỉ lệ)"""
        positions_all = np.vstack(self.pso.positions_history)  # gom toàn bộ vị trí lại để lấy min/max
        min_x, max_x = positions_all[:, 0].min(), positions_all[:, 0].max()
        min_y, max_y = positions_all[:, 1].min(), positions_all[:, 1].max()

        # Thêm khoảng cách đệm để tránh cắt biên
        pad_x = (max_x - min_x) * 0.1
        pad_y = (max_y - min_y) * 0.1
        x_min, x_max = min_x - pad_x, max_x + pad_x
        y_min, y_max = min_y - pad_y, max_y + pad_y

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')  # ✅ Tự động giữ tỉ lệ đúng giữa 2 trục

        ax.set_title(f"Tìm kiếm trên {self.name}")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        # Vẽ contour của hàm mục tiêu
        x = np.linspace(x_min, x_max, 150)
        y = np.linspace(y_min, y_max, 150)
        X, Y = np.meshgrid(x, y)
        Z = self.func([X, Y])
        contour = ax.contourf(X, Y, Z, levels=60, cmap='viridis', alpha=0.8)
        fig.colorbar(contour, label='Fitness Value $f(x)$')

        # Vẽ global minimum
        global_min_pos = get_global_min_pos(self.func)
        ax.plot(global_min_pos[0], global_min_pos[1], 'w*', markersize=12,
                markeredgecolor='black', label='Global Minimum')

        # Khởi tạo scatter plot cho particles
        scat = ax.scatter([], [], c='yellow', s=50, edgecolor='black', label='Particles')
        best_point, = ax.plot([], [], 'o', color='red', markersize=10,
                              markeredgecolor='black', label='Best Solution Found')
        ax.legend(loc='upper right')

        def update(frame):
            positions = self.pso.positions_history[frame]
            scat.set_offsets(positions)

            fitness = np.array([self.func(p) for p in positions])
            current_best = positions[np.argmin(fitness)]
            best_point.set_data([current_best[0]], [current_best[1]])

            ax.set_title(f'{self.name} PSO: Iteration {frame + 1}/{self.max_iter}')
            return scat, best_point

        anim = animation.FuncAnimation(
            fig, update, frames=len(self.pso.positions_history),
            interval=150, repeat=False, blit=False
        )

        gif_filename = f'pso_{self.name.lower()}_2d.gif'
        anim.save(gif_filename, writer='pillow', fps=10)
        plt.show()

    def plot_3d_surface(self):
        """Vẽ đồ thị 3D của hàm fitness và global minimum (tự động scale)"""
        x = np.linspace(self.lower_bound, self.upper_bound, 400)
        y = np.linspace(self.lower_bound, self.upper_bound, 400)
        X, Y = np.meshgrid(x, y)
        Z = self.func([X, Y])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.85)

        # Global minimum
        global_min_pos = get_global_min_pos(self.func)
        global_min_val = self.func(global_min_pos)
        ax.scatter(global_min_pos[0], global_min_pos[1], global_min_val,
                   color='red', marker='*', s=200, label='Global Minimum', zorder=5)

        # Giới hạn trục và scale hợp lý
        ax.set_xlim(self.lower_bound, self.upper_bound)
        ax.set_ylim(self.lower_bound, self.upper_bound)
        ax.set_zlim(Z.min(), Z.max())

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(f'Fitness Value $f(x)$ ({self.name})')
        ax.set_title(f"3D Surface of {self.name} Function")
        ax.legend()
        plt.tight_layout()
        plt.show()

# ====== 3 THUẬT TOÁN CỔ ĐIỂN ======
def hill_climbing(f, bounds, max_iter=100, step_size=0.05):
    """Thuật toán leo đồi (Hill Climbing) đơn giản"""
    x = np.random.uniform(bounds[0], bounds[1], 2)
    fx = f(x)
    history = [fx]

    for _ in range(max_iter):
        # Sinh ứng viên lân cận ngẫu nhiên
        neighbor = x + np.random.uniform(-step_size, step_size, 2)
        neighbor = np.clip(neighbor, bounds[0], bounds[1])
        f_neighbor = f(neighbor)

        # Nếu tốt hơn thì chấp nhận
        if f_neighbor < fx:
            x, fx = neighbor, f_neighbor
        history.append(fx)

    return x, fx, history

def simulated_annealing(f, bounds, max_iter=100, T0=1.0, alpha=0.95):
    x = np.random.uniform(bounds[0], bounds[1], 2)
    fx = f(x)
    history = [fx]
    T = T0
    for _ in range(max_iter):
        new_x = x + np.random.uniform(-1, 1, 2) * 0.1
        new_x = np.clip(new_x, bounds[0], bounds[1])
        new_fx = f(new_x)
        if new_fx < fx or np.random.rand() < np.exp((fx - new_fx) / T):
            x, fx = new_x, new_fx
        history.append(fx)
        T *= alpha
    return x, fx, history


def genetic_algorithm(f, bounds, pop_size=30, generations=100, mutation_rate=0.1):
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
    history = []

    def crossover(p1, p2):
        alpha = np.random.rand()
        return alpha * p1 + (1 - alpha) * p2

    for _ in range(generations):
        fitness = np.array([f(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        history.append(fitness[best_idx])

        selected = pop[np.argsort(fitness)[:pop_size // 2]]
        new_pop = []
        for _ in range(pop_size):
            p1, p2 = selected[np.random.randint(len(selected), size=2)]
            child = crossover(p1, p2)
            if np.random.rand() < mutation_rate:
                child += np.random.uniform(-0.1, 0.1, 2)
            new_pop.append(np.clip(child, bounds[0], bounds[1]))
        pop = np.array(new_pop)
    return pop[best_idx], fitness[best_idx], history


# ====== HÀM SO SÁNH 4 THUẬT TOÁN ======
def compare_classic_optimizers(f, bounds, max_iter=100):
    """So sánh PSO, Hill Climbing, SA, GA trên cùng một hàm"""
    print(f"\n=== Comparing Optimizers on {f.__name__.replace('_function', '').title()} ===")

    # === PSO ===
    pso = ParticleSwarmOptimization(f, bounds[0], bounds[1], max_iterations=max_iter)
    _, _, _ = pso.run()
    pso_best_curve = [f(pso.gbest_pos)] * max_iter

    # === Hill Climbing ===
    _, hc_val, hc_history = hill_climbing(f, bounds, max_iter=max_iter)

    # === Simulated Annealing ===
    _, sa_val, sa_history = simulated_annealing(f, bounds, max_iter=max_iter)

    # === Genetic Algorithm ===
    _, ga_val, ga_history = genetic_algorithm(f, bounds, generations=max_iter)

    # === VẼ SO SÁNH ===
    plt.figure(figsize=(8, 5))
    plt.plot(hc_history, label='Hill Climbing', color='orange')
    plt.plot(sa_history, label='Simulated Annealing', color='green')
    plt.plot(ga_history, label='Genetic Algorithm', color='purple')
    plt.plot(pso_best_curve, label='PSO (Final Value)', linestyle='--', color='red')

    plt.title(f"So sánh tốc độ hội tụ – {f.__name__.replace('_function', '').title()}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_and_compare():
    """Chạy và so sánh PSO với 3 thuật toán cổ điển"""
    max_iters = 50
    func_list = [ackley_function, rastrigin_function, rosenbrock_function, sphere_function]

    for f in func_list:
        l_bound, u_bound = get_function_bounds(f)

        print(f"\n{f.__name__.replace('_function', '').title()} Function")
        print(f"Bounds: [{l_bound}, {u_bound}]")

        # 1️⃣ Chạy PSO
        solver = ParticleSwarmOptimization(
            fitness_func=f,
            lower_bound=l_bound,
            upper_bound=u_bound,
            max_iterations=max_iters
        )
        _, best_pos, best_fit = solver.run()
        print(f"Best Fitness: {best_fit:.6f}, Best Position: {best_pos}")

        # 2️⃣ Hiển thị động học PSO
        vis = PSOVisualizer(solver)
        vis.animate_2d()
        vis.plot_3d_surface()

        # 3️⃣ So sánh với 3 thuật toán cổ điển (Hill Climbing, SA, GA)
        compare_classic_optimizers(f, (l_bound, u_bound), max_iter=max_iters)

if __name__ == "__main__":
    run_and_compare()