import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fitness_function import *


# ==========================
# 1️⃣ Firefly Algorithm (FA)
# ==========================
class FireflyAlgorithm:
    def __init__(self, fitness_func, lower_bound, upper_bound,
                 n_fireflies=50, max_iterations=100,
                 alpha=0.2, beta0=1.0, gamma=1.0, d=2):
        self.fitness_func = fitness_func
        self.function_name = fitness_func.__name__.replace('_function', '').title()
        self.dim = d
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_fireflies = n_fireflies
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

        self.positions = None
        self.fitness = None
        self.best_pos = None
        self.best_val = np.inf
        self.positions_history = []
        self.best_history = []  # lưu best-so-far

    def _initialize_fireflies(self):
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound,
                                           (self.n_fireflies, self.dim))
        self.fitness = np.array([self.fitness_func(p) for p in self.positions])
        best_idx = np.argmin(self.fitness)
        self.best_pos = self.positions[best_idx].copy()
        self.best_val = self.fitness[best_idx]
        self.best_history.append(self.best_val)

    def _update_fireflies(self):
        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                if self.fitness[j] < self.fitness[i]:
                    r = np.linalg.norm(self.positions[i] - self.positions[j])
                    beta = self.beta0 * np.exp(-self.gamma * r**2)
                    self.positions[i] += beta * (self.positions[j] - self.positions[i]) \
                                         + self.alpha * (np.random.rand(self.dim) - 0.5)
            # Clip vào bounds
            self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        # Cập nhật fitness
        self.fitness = np.array([self.fitness_func(p) for p in self.positions])
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_val:
            self.best_val = self.fitness[best_idx]
            self.best_pos = self.positions[best_idx].copy()

        # Lưu lịch sử best-so-far
        self.best_history.append(self.best_val)

    def run(self):
        self._initialize_fireflies()
        for _ in range(self.max_iterations):
            self.positions_history.append(np.copy(self.positions))
            self._update_fireflies()
        self.positions_history.append(np.copy(self.positions))
        return self.positions_history, self.best_pos, self.best_val, self.best_history


# ==========================
# 2️⃣ FA Visualizer
# ==========================
class FAVisualizer:
    def __init__(self, fa_instance):
        self.fa = fa_instance
        self.func = fa_instance.fitness_func
        self.name = fa_instance.function_name
        self.lower_bound = fa_instance.lower_bound
        self.upper_bound = fa_instance.upper_bound
        self.max_iter = fa_instance.max_iterations

    def animate_2d(self, save=False, filename=None):
        positions_history = self.fa.positions_history
        all_positions = np.concatenate(positions_history, axis=0)
        min_x, max_x = all_positions[:, 0].min(), all_positions[:, 0].max()
        min_y, max_y = all_positions[:, 1].min(), all_positions[:, 1].max()
        pad_x, pad_y = (max_x - min_x) * 0.1, (max_y - min_y) * 0.1
        x_min, x_max = min_x - pad_x, max_x + pad_x
        y_min, y_max = min_y - pad_y, max_y + pad_y

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"Tìm kiếm trên {self.name} (FA)")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        x = np.linspace(x_min, x_max, 150)
        y = np.linspace(y_min, y_max, 150)
        X, Y = np.meshgrid(x, y)
        Z = self.func([X, Y])
        contour = ax.contourf(X, Y, Z, levels=60, cmap="plasma", alpha=0.8)
        fig.colorbar(contour, label="Fitness Value $f(x)$")

        global_min_pos = get_global_min_pos(self.func)
        ax.plot(global_min_pos[0], global_min_pos[1], "w*", markersize=12,
                markeredgecolor="black", label="Global Minimum")

        scat = ax.scatter(positions_history[0][:, 0], positions_history[0][:, 1],
                          c="orange", s=50, edgecolor="black", label="Fireflies")
        best_point, = ax.plot([], [], "o", color="red", markersize=10,
                              markeredgecolor="black", label="Best Found")
        ax.legend(loc="upper right")

        def update(frame):
            pos = positions_history[frame]
            scat.set_offsets(pos)
            best_point.set_data(self.fa.best_history[frame], self.fa.best_history[frame])
            return scat, best_point

        ani = animation.FuncAnimation(fig, update, frames=len(positions_history),
                                      interval=150, repeat=False, blit=False)

        if save:
            if filename is None:
                filename = f"{self.name}_FA.gif"
            ani.save(filename, writer='pillow')
        plt.show()

    def plot_3d_surface(self):
        x = np.linspace(self.lower_bound, self.upper_bound, 400)
        y = np.linspace(self.lower_bound, self.upper_bound, 400)
        X, Y = np.meshgrid(x, y)
        Z = self.func([X, Y])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.85)

        global_min_pos = get_global_min_pos(self.func)
        global_min_val = self.func(global_min_pos)
        ax.scatter(global_min_pos[0], global_min_pos[1], global_min_val,
                   color='red', marker='*', s=200, label='Global Minimum', zorder=5)

        ax.set_xlim(self.lower_bound, self.upper_bound)
        ax.set_ylim(self.lower_bound, self.upper_bound)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(f'Fitness Value $f(x)$ ({self.name})')
        ax.set_title(f"3D Surface of {self.name} Function (FA)")
        ax.legend()
        plt.tight_layout()
        plt.show()


# ==========================
# 3️⃣ FA Parameter Sensitivity
# ==========================
def fa_parameter_sensitivity_analysis(f, bounds, max_iter=50):
    print(f"\n=== Phân tích độ nhạy tham số FA trên {f.__name__.replace('_function', '').title()} ===")
    param_ranges = {
        'alpha': np.linspace(0.1, 1.0, 6),
        'beta0': np.linspace(0.5, 2.0, 6),
        'gamma': np.linspace(0.1, 2.0, 6)
    }

    sensitivity_results = {}
    for param_name, values in param_ranges.items():
        best_vals = []
        for v in values:
            params = {'alpha': 0.2, 'beta0': 1.0, 'gamma': 1.0}
            params[param_name] = v
            fa = FireflyAlgorithm(f, bounds[0], bounds[1],
                                  max_iterations=max_iter,
                                  alpha=params['alpha'],
                                  beta0=params['beta0'],
                                  gamma=params['gamma'])
            fa.run()
            best_vals.append(fa.best_val)
        sensitivity_results[param_name] = (values, best_vals)

    plt.figure(figsize=(10, 5))
    for param_name, (values, bests) in sensitivity_results.items():
        plt.plot(values, bests, marker='o', linewidth=2, label=f'Ảnh hưởng {param_name}')
    plt.title(f"Độ nhạy tham số của FA – {f.__name__.replace('_function', '').title()}")
    plt.xlabel("Giá trị tham số")
    plt.ylabel("Fitness tốt nhất")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================
# Hill Climbing (best-so-far)
# ==========================
def hill_climbing(f, bounds, max_iter=100, step_size=0.05):
    x = np.random.uniform(bounds[0], bounds[1], 2)
    fx = f(x)
    best_fx = fx
    history = [best_fx]

    for _ in range(max_iter):
        neighbor = x + np.random.uniform(-step_size, step_size, 2)
        neighbor = np.clip(neighbor, bounds[0], bounds[1])
        f_neighbor = f(neighbor)
        if f_neighbor < fx:
            x, fx = neighbor, f_neighbor
        best_fx = min(best_fx, fx)
        history.append(best_fx)
    return x, best_fx, history


# ==========================
# Simulated Annealing (best-so-far)
# ==========================
def simulated_annealing(f, bounds, max_iter=100, T0=1.0, alpha=0.95):
    x = np.random.uniform(bounds[0], bounds[1], 2)
    fx = f(x)
    best_fx = fx
    history = [best_fx]
    T = T0

    for _ in range(max_iter):
        new_x = x + np.random.uniform(-1, 1, 2) * 0.1
        new_x = np.clip(new_x, bounds[0], bounds[1])
        new_fx = f(new_x)
        if new_fx < fx or np.random.rand() < np.exp((fx - new_fx) / T):
            x, fx = new_x, new_fx
        best_fx = min(best_fx, fx)
        history.append(best_fx)
        T *= alpha
    return x, best_fx, history


# ==========================
# Genetic Algorithm (best-so-far)
# ==========================
def genetic_algorithm(f, bounds, pop_size=30, generations=100, mutation_rate=0.1):
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, 2))
    fitness = np.array([f(ind) for ind in pop])
    best_fx = np.min(fitness)
    history = [best_fx]

    def crossover(p1, p2):
        alpha = np.random.rand()
        return alpha * p1 + (1 - alpha) * p2

    for _ in range(generations):
        fitness = np.array([f(ind) for ind in pop])
        best_fx = min(best_fx, np.min(fitness))
        history.append(best_fx)
        selected = pop[np.argsort(fitness)[:pop_size // 2]]
        new_pop = []
        for _ in range(pop_size):
            p1, p2 = selected[np.random.randint(len(selected), size=2)]
            child = crossover(p1, p2)
            if np.random.rand() < mutation_rate:
                child += np.random.uniform(-0.1, 0.1, 2)
            new_pop.append(np.clip(child, bounds[0], bounds[1]))
        pop = np.array(new_pop)
    return pop[np.argmin(fitness)], best_fx, history


# +
# ==========================
# 4️⃣ Run & Compare (FA + Classic Algorithms)
# ==========================
def run_and_compare():
    max_iters = 80
    func_list = [ackley_function, rastrigin_function, rosenbrock_function, sphere_function]

    for f in func_list:
        l_bound, u_bound = get_function_bounds(f)
        print(f"\n=== {f.__name__.replace('_function', '').title()} Function ===")

        # ===== 1️⃣ Chạy Firefly Algorithm =====
        fa = FireflyAlgorithm(f, l_bound, u_bound, max_iterations=max_iters)
        _, fa_best_pos, fa_best_val, fa_best_curve = fa.run()

        print(f"FA Best Fitness: {fa_best_val:.6f}, Best Position: {fa_best_pos}")

        # ===== 2️⃣ Chạy các thuật toán cổ điển =====
        _, hc_val, hc_history = hill_climbing(f, (l_bound, u_bound), max_iter=max_iters)
        # Chuyển hc_history thành best-so-far
        hc_best_curve = np.minimum.accumulate(hc_history)

        _, sa_val, sa_history = simulated_annealing(f, (l_bound, u_bound), max_iter=max_iters)
        sa_best_curve = np.minimum.accumulate(sa_history)

        _, ga_val, ga_history = genetic_algorithm(f, (l_bound, u_bound), generations=max_iters)
        ga_best_curve = np.minimum.accumulate(ga_history)

        # ===== 3️⃣ Vẽ đồ thị tốc độ hội tụ =====
        plt.figure(figsize=(8, 5))
        plt.plot(fa_best_curve, label='Firefly Algorithm (FA)', color='red', linewidth=2)
        plt.plot(hc_best_curve, label='Hill Climbing', color='orange', linewidth=2)
        plt.plot(sa_best_curve, label='Simulated Annealing', color='green', linewidth=2)
        plt.plot(ga_best_curve, label='Genetic Algorithm', color='purple', linewidth=2)

        plt.title(f"So sánh tốc độ hội tụ – {f.__name__.replace('_function', '').title()}")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness Value (best-so-far)")
        plt.yscale('log')  # nếu muốn log scale
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ===== 4️⃣ Visualizer FA =====
        visualizer = FAVisualizer(fa)
        visualizer.plot_3d_surface()
        visualizer.animate_2d()

        # ===== 5️⃣ Phân tích độ nhạy tham số FA =====
        fa_parameter_sensitivity_analysis(f, (l_bound, u_bound), max_iter=max_iters)


if __name__ == "__main__":
    run_and_compare()
