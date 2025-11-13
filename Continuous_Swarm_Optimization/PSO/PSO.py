import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fitness_function import *  # Import các hàm benchmark: ackley_function, rastrigin_function, rosenbrock_function, sphere_function

# ==========================
# Particle Swarm Optimization
# ==========================
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
        self.positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.n_particles, self.dim))
        self.velocities = np.zeros_like(self.positions)
        self.pbest_pos = np.copy(self.positions)
        self.pbest_val = np.array([self.fitness_func(p) for p in self.positions])
        self.gbest_pos = self.positions[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)

    def _update_particles(self):
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
        self._initialize_swarm()
        for _ in range(self.max_iterations):
            self.positions_history.append(np.copy(self.positions))
            self._update_particles()
        self.positions_history.append(np.copy(self.positions))
        return self.positions_history, self.gbest_pos, self.gbest_val

# ==========================
# PSO Visualizer
# ==========================
class PSOVisualizer:
    def __init__(self, pso_instance):
        self.pso = pso_instance
        self.func = pso_instance.fitness_func
        self.name = pso_instance.function_name
        self.lower_bound = pso_instance.lower_bound
        self.upper_bound = pso_instance.upper_bound
        self.max_iter = pso_instance.max_iterations

    def animate_2d(self):
        positions_all = np.vstack(self.pso.positions_history)
        min_x, max_x = positions_all[:, 0].min(), positions_all[:, 0].max()
        min_y, max_y = positions_all[:, 1].min(), positions_all[:, 1].max()
        pad_x, pad_y = (max_x - min_x) * 0.1, (max_y - min_y) * 0.1
        x_min, x_max = min_x - pad_x, max_x + pad_x
        y_min, y_max = min_y - pad_y, max_y + pad_y

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Tìm kiếm trên {self.name}")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        # Contour plot
        x = np.linspace(x_min, x_max, 150)
        y = np.linspace(y_min, y_max, 150)
        X, Y = np.meshgrid(x, y)
        Z = self.func([X, Y])
        contour = ax.contourf(X, Y, Z, levels=60, cmap='viridis', alpha=0.8)
        fig.colorbar(contour, label='Fitness Value $f(x)$')

        # Scatter for particles
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

        anim = animation.FuncAnimation(fig, update, frames=len(self.pso.positions_history),
                                       interval=150, repeat=False, blit=False)
        gif_filename = f'pso_{self.name.lower()}_2d.gif'
        anim.save(gif_filename, writer='pillow', fps=10)
        plt.show()

    def plot_3d_surface(self):
        x = np.linspace(self.lower_bound, self.upper_bound, 400)
        y = np.linspace(self.lower_bound, self.upper_bound, 400)
        X, Y = np.meshgrid(x, y)
        Z = self.func([X, Y])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.85)

        global_min_val = np.min(Z)
        ax.scatter(self.pso.gbest_pos[0], self.pso.gbest_pos[1], global_min_val,
                   color='red', marker='*', s=200, label='Global Minimum', zorder=5)

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

# ==========================
# Compare all classic optimizers
# ==========================
def compare_classic_optimizers(f, bounds, max_iter=100):
    print(f"\n=== Comparing Optimizers on {f.__name__.replace('_function', '').title()} ===")
    seed = 0

    # PSO
    np.random.seed(seed)
    pso = ParticleSwarmOptimization(f, bounds[0], bounds[1], max_iterations=max_iter)
    _, _, _ = pso.run()
    pso_best_curve = []
    best_so_far = np.inf
    for pos in pso.positions_history:
        cur_best = np.min([f(p) for p in pos])
        best_so_far = min(best_so_far, cur_best)
        pso_best_curve.append(best_so_far)

    # Hill Climbing
    np.random.seed(seed)
    _, _, hc_history = hill_climbing(f, bounds, max_iter=max_iter)

    # Simulated Annealing
    np.random.seed(seed)
    _, _, sa_history = simulated_annealing(f, bounds, max_iter=max_iter)

    # Genetic Algorithm
    np.random.seed(seed)
    _, _, ga_history = genetic_algorithm(f, bounds, generations=max_iter)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(hc_history, label='Hill Climbing', linewidth=2)
    plt.plot(sa_history, label='Simulated Annealing', linewidth=2)
    plt.plot(ga_history, label='Genetic Algorithm', linewidth=2)
    plt.plot(pso_best_curve, label='PSO', linestyle='--', linewidth=2)
    plt.title(f"So sánh tốc độ hội tụ – {f.__name__.replace('_function', '').title()}")
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==========================
# Parameter sensitivity analysis for PSO
# ==========================
def parameter_sensitivity_analysis(f, bounds, max_iter=50):
    print(f"\n=== Phân tích độ nhạy tham số PSO trên {f.__name__.replace('_function', '').title()} ===")
    param_ranges = {
        'w': np.linspace(0.4, 0.9, 6),
        'c1': np.linspace(0.5, 2.5, 6),
        'c2': np.linspace(0.5, 2.5, 6)
    }
    sensitivity_results = {}

    for param_name, values in param_ranges.items():
        best_values = []
        for val in values:
            kwargs = {'w': 0.7, 'c1': 1.5, 'c2': 1.5}
            kwargs[param_name] = val
            pso = ParticleSwarmOptimization(f, bounds[0], bounds[1],
                                            max_iterations=max_iter, **kwargs)
            _, _, _ = pso.run()
            best_values.append(pso.gbest_val)
        sensitivity_results[param_name] = (values, best_values)

    plt.figure(figsize=(10, 5))
    for param_name, (values, best_vals) in sensitivity_results.items():
        plt.plot(values, best_vals, marker='o', label=f'{param_name} sensitivity')
    plt.title(f"Phân tích độ nhạy tham số PSO – {f.__name__.replace('_function', '').title()}")
    plt.xlabel("Giá trị tham số")
    plt.ylabel("Giá trị fitness tốt nhất")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_and_compare()


# ==========================
# Run and compare all
# ==========================
def run_and_compare():
    max_iters = 50
    func_list = [ackley_function, rastrigin_function, rosenbrock_function, sphere_function]

    for f in func_list:
        l_bound, u_bound = get_function_bounds(f)
        print(f"\n{f.__name__.replace('_function', '').title()} Function")
        print(f"Bounds: [{l_bound}, {u_bound}]")

        # PSO solver
        solver = ParticleSwarmOptimization(f, l_bound, u_bound, max_iterations=max_iters)
        _, best_pos, best_fit = solver.run()
        print(f"Best Fitness: {best_fit:.6f}, Best Position: {best_pos}")

        # Visualize PSO
        vis = PSOVisualizer(solver)
        vis.animate_2d()
        vis.plot_3d_surface()

        # Compare classic optimizers
        compare_classic_optimizers(f, (l_bound, u_bound), max_iter=max_iters)

        # Parameter sensitivity
        parameter_sensitivity_analysis(f, (l_bound, u_bound), max_iter=max_iters)
