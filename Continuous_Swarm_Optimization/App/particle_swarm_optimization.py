import numpy as np

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
        
        self.positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.n_particles, self.dim))
        self.velocities = np.zeros_like(self.positions)
        self.pbest_pos = np.copy(self.positions)
        
        self.pbest_val = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            self.pbest_val[i] = self.fitness_func(self.positions[i])
            self.nfe += 1 

        self.gbest_pos = self.positions[np.argmin(self.pbest_val)].copy()
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

        
        current_vals = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            if self.nfe >= self.max_nfe and self.max_nfe > 0:
                current_vals[i] = float('inf')
                continue
            current_vals[i] = self.fitness_func(self.positions[i])
            self.nfe += 1 

        
        better = current_vals < self.pbest_val
        self.pbest_pos[better] = self.positions[better]
        self.pbest_val[better] = current_vals[better]

        
        current_best_val = np.min(current_vals)
        if current_best_val < self.gbest_val:
            self.gbest_val = current_best_val
            self.gbest_pos = self.positions[np.argmin(current_vals)].copy()

    def run_by_iteration(self):
        
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
        
        self._initialize_swarm() 
        self.history_best_fitness = [] 
        self.positions_history = []
        
        
        for _ in range(self.max_iterations):
            if self.nfe >= self.max_nfe:
                break
                
            self.positions_history.append(np.copy(self.positions))
            self._update_particles() 
            self.history_best_fitness.append(self.gbest_val) 

        self.positions_history.append(np.copy(self.positions))
        return self.positions_history, self.gbest_pos, self.gbest_val, self.history_best_fitness