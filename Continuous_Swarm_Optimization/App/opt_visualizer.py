import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fitness_function import *
class OptimizationVisualizer:
    """Visualizer"""

    def __init__(self, instance=None, fitness_func=None, lower_bound=None, upper_bound=None):
        self.instance = instance
        if not instance and fitness_func and lower_bound and upper_bound:
            self.func = fitness_func
            self.name = fitness_func.__name__.replace('_function', '').title()
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
        else:
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

    def plot_3d_surface(self):
        """Vẽ đồ thị 3D."""
        x = np.linspace(self.lower_bound, self.upper_bound, 500)
        y = np.linspace(self.lower_bound, self.upper_bound, 500)
        x_meshgrid, y_meshgrid = np.meshgrid(x, y)
        z = self.func([x_meshgrid, y_meshgrid])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_meshgrid, y_meshgrid, z,
                        cmap='viridis', edgecolor='none', alpha=0.8)

        global_min_pos = get_global_min_pos(self.func)
        global_min_val = self.func(global_min_pos)
        ax.scatter(global_min_pos[0], global_min_pos[1], global_min_val,
                   color='red', marker='*', s=200, label='Global Minimum', zorder=5)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(f'Fitness Value $f(x)$ ({self.name})')
        ax.set_title(f"Đồ thị {self.name} Function")
        ax.legend()
        plt.tight_layout()
        plt.show()