### import tất cả các thư viện cần thiết
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['animation.embed_limit'] = 50
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import display

"""Các bài toán sẽ được áp dụng"""

### Các bài toán Sphere, Rastrigin, Rosenbrock và Ackley
class ObjectiveFunctions:
    """
    Tập hợp các hàm mục tiêu (Objective Functions)
    """
    @staticmethod
    def sphere_function(x):
        """Hàm Sphere: f(x) = sum(x^2). Global minimum tại x=0, f(x)=0."""
        return np.sum(x**2)

    @staticmethod
    def rastrigin_function(x):
        """Hàm Rastrigin: f(x) = A*d + sum(x^2 - A*cos(2*pi*x)). Global minimum tại x=0, f(x)=0."""
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    @staticmethod
    def rosenbrock_function(x):
        """Hàm Rosenbrock: f(x) = sum(100*(x[i+1]-x[i]^2)^2 + (x[i]-1)^2). Global minimum tại x=1, f(x)=0."""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

    @staticmethod
    def ackley_function(x):
        """Hàm Ackley: f(x) = -20*exp(-0.2*sqrt(1/n*sum(x^2))) - exp(1/n*sum(cos(2*pi*x))) + 20 + e.
        Global minimum tại x=0, f(x)=0."""
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

"""Thuật toán Cuckoo Search"""

### Cuckoo Search (CSAlgorithm)
class CSAlgorithm:
    def __init__(self, obj_func):
        self.obj_func = obj_func

    def _levy_flight(self, Lambda, dim):
        """Tính toán bước nhảy Levy Flight (Private method)."""
        sigma = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
                 (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / Lambda)
        return step

    def cuckoo_search(self, n=25, dim=2, lb=-10, ub=10,
                      pa=0.15, alpha=0.5, max_nfe=10000,
                      F=0.5, CR=0.9, levy_beta=1.5, seed=None,
                      record_history=False):
        
        if seed is not None:
            np.random.seed(seed)

        nests = np.random.uniform(lb, ub, size=(n, dim))
        fitness = np.array([self.obj_func(x) for x in nests])
        nfe_count = n

        best_idx = np.argmin(fitness)
        best = nests[best_idx].copy()
        best_fitness = fitness[best_idx]

        nfe_history = [nfe_count]
        best_fitness_history = [best_fitness]
        positions_history = [] if record_history else None

        iter_count = 0
        while nfe_count < max_nfe:
            iter_count += 1

            if record_history:
                positions_history.append(nests.copy())

            # 1. Lévy flights
            for i in range(n):
                if nfe_count >= max_nfe: break
                step = self._levy_flight(levy_beta, dim)
                phi = np.random.rand()
                # Chọn ngẫu nhiên một tổ khác để tính toán
                random_nest_idx = np.random.choice(nests.shape[0])
                new_nest = nests[i] + alpha * step * (nests[i] - nests[random_nest_idx])
                # Sử dụng thông tin của best (tăng cường tìm kiếm cục bộ)
                new_nest += phi * (best - nests[i]) * 0.5
                new_nest = np.clip(new_nest, lb, ub)
                new_fit = self.obj_func(new_nest)
                nfe_count += 1

                if new_fit < fitness[i]:
                    nests[i] = new_nest
                    fitness[i] = new_fit

                    if new_fit < best_fitness:
                        best = new_nest.copy()
                        best_fitness = new_fit
                        nfe_history.append(nfe_count)
                        best_fitness_history.append(best_fitness)

            if nfe_count >= max_nfe: break

            # 2. Mixing phase
            perm = np.random.permutation(n)
            new_nests = nests.copy()
            for i in range(n):
                idxs = [idx for idx in range(n) if idx != i]
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = nests[i] + F * (nests[a] - nests[b])
                cross = np.random.rand(dim) < CR
                if not np.any(cross):
                    cross[np.random.randint(dim)] = True
                trial = np.where(cross, mutant, nests[i])
                new_nests[i] = np.clip(trial, lb, ub)

            new_fitness = np.array([self.obj_func(x) for x in new_nests])
            nfe_count += n

            improved = new_fitness < fitness
            fitness[improved] = new_fitness[improved]
            nests[improved] = new_nests[improved]

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best = nests[best_idx].copy()
                best_fitness = fitness[best_idx]
                nfe_history.append(nfe_count)
                best_fitness_history.append(best_fitness)

            if nfe_count >= max_nfe: break

            # 3. Discovery
            num_replace = max(1, int(pa * n))
            worst_idx = np.argsort(fitness)[-num_replace:]
            for idx in worst_idx:
                if np.random.rand() < 0.5:
                    nests[idx] = np.random.uniform(lb, ub, dim)
                else:
                    nests[idx] = np.clip(best + np.random.normal(0, 0.1 * (ub - lb), size=dim), lb, ub)

            fitness[worst_idx] = np.array([self.obj_func(x) for x in nests[worst_idx]])
            nfe_count += len(worst_idx)

            # 4. Tìm kiếm cục bộ định kỳ
            if iter_count % 10 == 0:
                local = best + np.random.normal(0, 0.01 * (ub - lb), size=(5, dim))
                local = np.clip(local, lb, ub)
                local_f = np.array([self.obj_func(x) for x in local])
                nfe_count += local.shape[0]
                mi = np.argmin(local_f)
                if local_f[mi] < best_fitness:
                    best = local[mi].copy()
                    best_fitness = local_f[mi]
                    worst = np.argmax(fitness)
                    nests[worst] = local[mi].copy()
                    fitness[worst] = local_f[mi]
                    nfe_history.append(nfe_count)
                    best_fitness_history.append(best_fitness)

            nfe_history.append(nfe_count)
            best_fitness_history.append(best_fitness)

        if record_history:
            positions_history.append(nests.copy())

        if record_history:
            return best, best_fitness, nfe_history, best_fitness_history, positions_history
        else:
            return best, best_fitness, nfe_history, best_fitness_history

"""Các thuật toán cổ điển khác"""

### Các thuật toán tìm kiếm cổ điển khác
class ClassicAlgorithms:
    """
    Tập hợp các thuật toán tối ưu hóa cổ điển (HC, SA, GA).
    Mỗi phương thức sẽ trả về 4 giá trị: (best, best_fitness, nfe_history, best_fitness_history)
    """
    def __init__(self, obj_func):
        self.obj_func = obj_func

    def hill_climbing(self, dim=2, lb=-10, ub=10, step_size=0.1, max_nfe=10000, num_neighbors=20):
        current = np.random.uniform(lb, ub, dim)
        current_fitness = self.obj_func(current)

        nfe_count = 1
        nfe_history = [nfe_count]
        best_fitness_history = [current_fitness]

        while nfe_count < max_nfe:
            neighbors = [current + np.random.uniform(-step_size, step_size, dim) for _ in range(num_neighbors)]
            neighbors = [np.clip(n, lb, ub) for n in neighbors]

            neighbor_fitness = []
            for n in neighbors:
                if nfe_count >= max_nfe: break
                neighbor_fitness.append(self.obj_func(n))
                nfe_count += 1

            if not neighbor_fitness: break

            best_neighbor_idx = np.argmin(neighbor_fitness)
            best_neighbor = neighbors[best_neighbor_idx]
            best_neighbor_fitness = neighbor_fitness[best_neighbor_idx]

            if best_neighbor_fitness < current_fitness:
                current, current_fitness = best_neighbor, best_neighbor_fitness

            nfe_history.append(nfe_count)
            best_fitness_history.append(current_fitness)

        return current, current_fitness, nfe_history, best_fitness_history

    def simulated_annealing(self, dim=2, lb=-10, ub=10, T0=100, cooling_rate=0.99, max_nfe=10000):
        current = np.random.uniform(lb, ub, dim)
        current_fitness = self.obj_func(current)
        best_fitness = current_fitness

        nfe_count = 1
        nfe_history = [nfe_count]
        best_fitness_history = [best_fitness]

        T = T0
        while nfe_count < max_nfe:
            candidate = current + np.random.uniform(-1, 1, dim)
            candidate = np.clip(candidate, lb, ub)
            candidate_fitness = self.obj_func(candidate)
            nfe_count += 1

            delta = candidate_fitness - current_fitness
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current, current_fitness = candidate, candidate_fitness

            if current_fitness < best_fitness:
                best_fitness = current_fitness

            T *= cooling_rate

            nfe_history.append(nfe_count)
            best_fitness_history.append(best_fitness)

        return current, best_fitness, nfe_history, best_fitness_history

    def genetic_algorithm(self, dim=2, lb=-10, ub=10, pop_size=40, crossover_rate=0.8, mutation_rate=0.1, max_nfe=10000):
        population = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.array([self.obj_func(ind) for ind in population])

        nfe_count = pop_size
        best_fitness = np.min(fitness)

        nfe_history = [nfe_count]
        best_fitness_history = [best_fitness]

        while nfe_count < max_nfe:
            fitness_inv = 1 / (fitness + 1e-8)
            probs = fitness_inv / np.sum(fitness_inv)
            parents_idx = np.random.choice(pop_size, size=pop_size, p=probs)
            parents = population[parents_idx]

            offspring = []
            for i in range(0, pop_size, 2):
                p1, p2 = parents[i], parents[(i + 1) % pop_size]
                if np.random.rand() < crossover_rate:
                     cross_point = np.random.randint(1, dim)
                     child1 = np.concatenate([p1[:cross_point], p2[cross_point:]])
                     child2 = np.concatenate([p2[:cross_point], p1[cross_point:]])
                else:
                     child1, child2 = p1.copy(), p2.copy()
                offspring.extend([child1, child2])

            for i in range(len(offspring)):
                if np.random.rand() < mutation_rate:
                    mutate_dim = np.random.randint(0, dim)
                    offspring[i][mutate_dim] += np.random.uniform(-1, 1)
                    offspring[i] = np.clip(offspring[i], lb, ub)

            new_population = np.array(offspring)[:pop_size]

            new_fitness = []
            for ind in new_population:
                 if nfe_count >= max_nfe: break
                 new_fitness.append(self.obj_func(ind))
                 nfe_count += 1

            if not new_fitness: break

            population = new_population[:len(new_fitness)]
            fitness = np.array(new_fitness)

            current_best = np.min(fitness)
            if current_best < best_fitness:
                best_fitness = current_best

            nfe_history.append(nfe_count)
            best_fitness_history.append(best_fitness)

        return population[np.argmin(fitness)], best_fitness, nfe_history, best_fitness_history

"""Các class và function hỗ trợ"""

### Visualization
class OptimizationVisualizer:
    """Visualization quá trình tối ưu hóa (animation + 2D contour)."""
    def __init__(self, func, lb, ub, func_name=None):
        self.func = func
        self.lb = lb
        self.ub = ub
        self.name = func_name or func.__name__.replace('_function', '').title()

    def animate_optimization_2d(self, positions_history, interval=150):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(self.lb, self.ub)
        ax.set_ylim(self.lb, self.ub)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(f"Tối ưu hóa {self.name}", fontsize=13)

        X = np.linspace(self.lb, self.ub, 200)
        Y = np.linspace(self.lb, self.ub, 200)
        X, Y = np.meshgrid(X, Y)
        Z = np.array([self.func(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
        ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)

        scat = ax.scatter([], [], c='yellow', s=60, edgecolors='k', label='Nests')
        best, = ax.plot([], [], 'ro', markersize=8, label='Best')
        ax.legend(loc='upper right')

        def update(frame):
            nests = positions_history[frame]
            fits = np.array([self.func(p) for p in nests])
            best_pos = nests[np.argmin(fits)]
            scat.set_offsets(nests)
            best.set_data([best_pos[0]], [best_pos[1]])
            ax.set_title(f'{self.name} - Iteration {frame+1}/{len(positions_history)}')
            return scat, best

        anim = animation.FuncAnimation(
            fig, update, frames=len(positions_history),
           interval=interval, blit=False, repeat=False
        )

        plt.show()

### Biểu đồ hội tụ
def plot_convergence(nfe_history, best_fitness_history,
                     title="Convergence Curve", save_path=None):

    epsilon = np.finfo(float).eps
    y_values = np.array(best_fitness_history)
    y_values[y_values <= 0] = epsilon

    plt.figure(figsize=(8, 5))
    plt.plot(nfe_history, y_values, color='blue', linewidth=2)
    plt.xlabel('Number of Function Evaluations (NFE)', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.yscale('log')

    min_val = min(y_values)
    max_val = max(y_values)

    bottom_limit = min_val * 0.5

    plt.ylim(bottom=bottom_limit, top=max_val * 1.1)

    plt.title(title, fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

### Biểu đồ bề mặt 3D
def plot_3D_surface(func, lb=-5, ub=5, resolution=100, title="Objective Function Surface"):
    X = np.linspace(lb, ub, resolution)
    Y = np.linspace(lb, ub, resolution)
    X, Y = np.meshgrid(X, Y)

    Z = np.array([func(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('f(X,Y)', fontsize=11)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_parameter_sensitivity(
    obj_func, algorithm_factory, param_name,
    param_values, func_display_name,
    n=25, dim=2, lb=-10, ub=10, 
    max_nfe=5000, repeats=20, base_seed=1, 
    fixed_pa=None, fixed_alpha=None):

    mean_best_fitness = []

    default_params = {
        'pa': 0.15 if fixed_pa is None else fixed_pa,
        'alpha': 0.5 if fixed_alpha is None else fixed_alpha,
        'levy_beta': 1.5,
        'n': n, 'dim': dim,
        'lb': lb, 'ub': ub, 'max_nfe': max_nfe
    }

    class TempCS(algorithm_factory):
        def __init__(self, obj_func, **kwargs):
            super().__init__(obj_func)
            self.kwargs = kwargs
        def run(self):
            return self.cuckoo_search(record_history=False, **self.kwargs)

    print(f"Bắt đầu phân tích độ nhạy {param_name.upper()} trên {func_display_name} (Repeats={repeats})")

    for val in param_values:
        params = default_params.copy()
        params[param_name] = val

        run_results = []
        for r in range(repeats):
            current_params = params.copy()
            current_params['seed'] = base_seed + r

            _, best_fit, _, _ = TempCS(obj_func, **current_params).run()
            run_results.append(best_fit)

        mean_fit = np.mean(run_results)
        mean_best_fitness.append(mean_fit)
        print(f"   {param_name}={val}: Mean Fitness = {mean_fit:.4e}")

    data_points = list(zip(param_values, mean_best_fitness))
    data_points.sort(key=lambda x: x[0])
    sorted_param_values, sorted_mean_best_fitness = zip(*data_points)
    processed_fitness = np.array(sorted_mean_best_fitness, dtype=float)

    # Tự động chọn scale: linear nếu có giá trị <= 0, log nếu tất cả > 0
    # Xử lý: Nếu min_fit quá nhỏ (<= 1e-10), có thể có lỗi log.
    use_log = np.all(processed_fitness > 0)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_param_values, processed_fitness, marker='o', linewidth=2, color='red')
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel(f'Mean Best Fitness (over {repeats} runs)', fontsize=12)

    # Đảm bảo scale được chọn phù hợp
    if use_log and np.min(processed_fitness) > 1e-10:
        plt.yscale('log')
    else:
        # Nếu không thể dùng log (do giá trị quá nhỏ hoặc âm), dùng linear
        plt.yscale('linear')
        print("Cảnh báo: Không sử dụng thang log vì kết quả quá gần hoặc bằng 0.")

    fixed_info = ""
    if param_name != 'pa':
        fixed_info += f" | Fixed pa={default_params['pa']}"
    if param_name != 'alpha':
        fixed_info += f" | Fixed alpha={default_params['alpha']}"

    plt.title(f'Sensitivity Analysis of {param_name.upper()} on {func_display_name}{fixed_info}', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

"""Hàm hỗ trợ việc so sánh thuật toán CS với các thuật toán khác"""

### Hàm hỗ trợ so sánh CS với các thuật toán khác
def compare_algorithms(objective_function, algorithms, dim=2, lb=-5.12, ub=5.12,
                       max_nfe=5000, repeats=1, base_seed=1): 
    # ... (Giữ nguyên x_grid và eps)
    x_grid = np.arange(max_nfe + 1)
    eps = 1e-12

    plt.figure(figsize=(10,6))

    for name, factory in algorithms.items():
        curves = []
        for r in range(repeats):
            np.random.seed(base_seed + r)
            _, _, nfe_hist, best_hist = factory()
            y = np.full(max_nfe+1, np.nan)
            nfe_hist = np.array(nfe_hist)
            best_hist = np.array(best_hist)
            idx = 0
            current_best = np.nan
            for i, g in enumerate(x_grid):
                while idx < len(nfe_hist) and nfe_hist[idx] <= g:
                    current_best = best_hist[idx]
                    idx += 1
                y[i] = current_best

            first_valid = np.where(~np.isnan(y))[0][0]
            y[:first_valid] = y[first_valid]
            y[np.isnan(y)] = y[~np.isnan(y)][-1]
            curves.append(y)

        curves = np.array(curves)
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        plt.plot(x_grid, mean_curve, label=name, linewidth=2)
        plt.fill_between(x_grid, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.xlabel('Number of Function Evaluations (NFE)', fontsize=12)
    plt.ylabel('Best Fitness (log scale)', fontsize=12)
    plt.yscale('log')
    plt.title(f'Algorithm Comparison on {objective_function.__name__}', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()

"""Thực hiện yêu cầu bài toán"""

### Thực hiện chạy thuật toán Cuckoo, lập các biểu đồ và so sánh với các thuật toán tìm kiếm khác theo yêu cầu của đề
class ExperimentRunner:
    """
    Chạy các thí nghiệm, so sánh và trực quan hóa kết quả.
    """
    def __init__(self, obj_func, func_name, lb, ub, dim=2, max_nfe=5000, seed=1, pa = 0.25, alpha=0.5):
        self.obj_func = obj_func
        self.func_name = func_name
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.max_nfe = max_nfe
        self.seed = seed
        self.pa = pa
        self.alpha = alpha

        self.cs_solver = CSAlgorithm(obj_func)
        self.classic_solver = ClassicAlgorithms(obj_func)


    def run_cuckoo_search(self):
        """Chạy Cuckoo Search và hiển thị kết quả chi tiết."""
        print(f"--- Chạy Cuckoo Search trên {self.func_name} ---")

        best, best_fitness, nfe_hist, best_hist, positions_history = self.cs_solver.cuckoo_search(
            dim=self.dim, lb=self.lb, ub=self.ub, max_nfe=self.max_nfe, seed=self.seed, pa = self.pa, alpha=self.alpha, record_history=True
        )

        print(f"Best solution: {best}")
        print(f"Best fitness: {best_fitness}")

        plot_convergence(nfe_hist, best_hist, title=f"Cuckoo Search Convergence ({self.func_name})")

        visualizer = OptimizationVisualizer(self.obj_func, lb=self.lb, ub=self.ub, func_name=self.func_name)
        result = visualizer.animate_optimization_2d(positions_history)
        if result is not None:
             display(result)

        if self.dim == 2:
             plot_3D_surface(self.obj_func, lb=self.lb, ub=self.ub, title=f"{self.func_name} Surface")

    def analyze_parameter_sensitivity(self, repeats=5, sensitivity_max_nfe = 5000):
        """Thực hiện phân tích độ nhạy cho PA và ALPHA, chạy nhiều lần lặp."""

        print("\n--- Phân tích Độ nhạy của PA ---")
        plot_parameter_sensitivity(
            obj_func=self.obj_func,
            algorithm_factory=CSAlgorithm,
            param_name='pa',
            param_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
            repeats=repeats,
            fixed_alpha=self.alpha,
            func_display_name=self.func_name,
            dim=self.dim, lb=self.lb, ub=self.ub, max_nfe=sensitivity_max_nfe
        )

        print("\n--- Phân tích Độ nhạy của ALPHA ---")
        plot_parameter_sensitivity(
            obj_func=self.obj_func,
            algorithm_factory=CSAlgorithm,
            param_name='alpha',
            param_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            repeats=repeats,
            fixed_pa=self.pa,
            func_display_name=self.func_name,
            dim=self.dim, lb=self.lb, ub=self.ub, max_nfe=sensitivity_max_nfe
        )

    def compare_all_algorithms(self, repeats=1):
        """So sánh CS với các thuật toán cổ điển."""
        print(f"\n--- So sánh thuật toán trên {self.func_name} ---")

        algorithms = {
            "Cuckoo Search": lambda: self.cs_solver.cuckoo_search(
                dim=self.dim, lb=self.lb, ub=self.ub, max_nfe=self.max_nfe, seed=None,
                pa=self.pa, alpha=self.alpha)[:4],
            "Hill Climbing": lambda: self.classic_solver.hill_climbing(dim=self.dim, lb=self.lb, ub=self.ub, max_nfe=self.max_nfe),
            "Simulated Annealing": lambda: self.classic_solver.simulated_annealing(dim=self.dim, lb=self.lb, ub=self.ub, max_nfe=self.max_nfe),
            "Genetic Algorithm": lambda: self.classic_solver.genetic_algorithm(dim=self.dim, lb=self.lb, ub=self.ub, max_nfe=self.max_nfe),
        }

        compare_algorithms(self.obj_func, algorithms, dim=self.dim, lb=self.lb, ub=self.ub,
                           max_nfe=self.max_nfe, repeats=repeats, base_seed=self.seed)

def main():
    ### Mời thầy chọn từng bài toán để chạy ạ !!!

    ### Bài toán 1: Sphere Function
    print("=== SPHERE FUNCTION ===")
    sphere_runner = ExperimentRunner(
        obj_func=ObjectiveFunctions.sphere_function,
        func_name="Sphere Function",
        lb=-5.12, ub=5.12, dim=2, max_nfe=5000, seed=1,
        alpha=0.5, pa=0.25
    )
    sphere_runner.run_cuckoo_search()
    sphere_runner.compare_all_algorithms(repeats=1)
    sphere_runner.analyze_parameter_sensitivity(repeats=1)

    ### Bài toán 2: Rastrigin Function
    print("=== RASTRIGIN FUNCTION ===")
    rastrigin_runner = ExperimentRunner(
        obj_func=ObjectiveFunctions.rastrigin_function,
        func_name="Rastrigin Function",
        lb=-5.12, ub=5.12, dim=2, max_nfe=5000,
        seed=1, # Đặt seed khác biệt để đảm bảo tính lặp lại cho bài toán này
        alpha=0.5, pa=0.25
    )
    rastrigin_runner.run_cuckoo_search()
    rastrigin_runner.compare_all_algorithms(repeats=1)
    rastrigin_runner.analyze_parameter_sensitivity(repeats=1)

    ### Bài toán 3: Rosenbrock Function
    print("=== ROSENBROCK FUNCTION ===")
    rosenbrock_runner = ExperimentRunner(
        obj_func=ObjectiveFunctions.rosenbrock_function,
        func_name="Rosenbrock Function",
        lb=-5.12, ub=5.12, dim=2, max_nfe=5000,
        seed=1,
        alpha=0.5, pa=0.25
    )
    rosenbrock_runner.run_cuckoo_search()
    rosenbrock_runner.compare_all_algorithms(repeats=1)
    rosenbrock_runner.analyze_parameter_sensitivity(repeats=1)

    ### Bài toán 4: Ackley Function
    print("=== ACKLEY FUNCTION ===")
    ackley_runner = ExperimentRunner(
        obj_func=ObjectiveFunctions.ackley_function,
        func_name="Ackley Function",
        lb=-5.12, ub=5.12, dim=2, max_nfe=5000,
        seed=1,
        alpha=0.5, pa=0.25
    )
    ackley_runner.run_cuckoo_search()
    ackley_runner.compare_all_algorithms(repeats=1)
    ackley_runner.analyze_parameter_sensitivity(repeats=1)

if __name__ == "__main__":
    main()