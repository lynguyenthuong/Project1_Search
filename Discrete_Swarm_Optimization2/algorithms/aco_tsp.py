import numpy as np
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from problems.tsp import tour_length

Tour = List[int]
History = List[float]
AlgorithmResult = Tuple[Tour, float, History]
IterationCallback = Optional[Callable[[int, float, Optional[Tour], Dict[str, np.ndarray]], None]]

@dataclass
class ACOConfig:
    """Hyper-parameters for the Ant Colony Optimization solver."""

    n_ants: Optional[int] = None
    n_iterations: int = 200
    alpha: float = 1.0
    beta: float = 5.0
    rho: float = 0.45        # evaporation rate
    q: float = 75.0          # pheromone deposit factor
    elitist_weight: float = 0.5  # >0 để tăng cường best_global
    seed: int = 42


class AntColonyTSP:
    """Classic Ant Colony Optimization solver for the symmetric TSP."""

    def __init__(self, D: np.ndarray, cfg: ACOConfig):
        self.D = D.astype(float)
        self.n = D.shape[0]
        self.cfg = cfg
        # n_ants động nếu chưa set
        if self.cfg.n_ants is None:
            self.cfg.n_ants = self.n if self.n <= 20 else max(10, self.n // 3)

        self.rng = np.random.default_rng(cfg.seed)
        self.tau = np.ones((self.n, self.n), dtype=float)            # pheromone
        self.eta = 1.0 / (self.D + 1e-12)                            # heuristic
        np.fill_diagonal(self.eta, 0.0)

        self.best_tour: Optional[Tour] = None
        self.best_length: float = float("inf")

        # Ngưỡng clamp pheromone để tránh bùng nổ / tiêu biến
        self._tau_min, self._tau_max = 1e-9, 1e9

    def _select_next(self, i: int, unvisited) -> int:
        unv = np.fromiter(unvisited, dtype=int)
        numer = (self.tau[i, unv] ** self.cfg.alpha) * (self.eta[i, unv] ** self.cfg.beta)
        s = float(numer.sum())
        if s <= 0.0 or not np.isfinite(s):
            # fallback uniform
            return int(self.rng.choice(unv))
        probs = numer / s
        return int(self.rng.choice(unv, p=probs))

    def _build_tour(self, start: int) -> List[int]:
        tour = [start]
        unvisited = set(range(self.n))
        unvisited.remove(start)
        while unvisited:
            j = self._select_next(tour[-1], unvisited)
            tour.append(j)
            unvisited.remove(j)
        tour.append(start)
        return tour

    def _update_pheromones(self, tours: List[List[int]], lengths: List[float]) -> None:
        # Bay hơi
        self.tau *= (1.0 - self.cfg.rho)

        # Lắng đọng từ tất cả kiến
        for tour, L in zip(tours, lengths):
            dep = self.cfg.q / (L + 1e-12)
            for k in range(len(tour) - 1):
                a, b = tour[k], tour[k + 1]
                self.tau[a, b] += dep
                self.tau[b, a] += dep

        # Elitist: tăng cho best toàn cục (nếu bật)
        if self.cfg.elitist_weight > 0 and self.best_tour is not None:
            dep = self.cfg.elitist_weight * self.cfg.q / (self.best_length + 1e-12)
            for k in range(len(self.best_tour) - 1):
                a, b = self.best_tour[k], self.best_tour[k + 1]
                self.tau[a, b] += dep
                self.tau[b, a] += dep

        # Clamp để ổn định số học
        np.clip(self.tau, self._tau_min, self._tau_max, out=self.tau)

    def run(self, on_iter: IterationCallback = None) -> AlgorithmResult:
        n_ants = int(self.cfg.n_ants)
        history: History = []

        for it in range(self.cfg.n_iterations):
            tours: List[List[int]] = []
            lengths: List[float] = []

            # Mẹo: xoay vòng điểm start để đa dạng
            for k in range(n_ants):
                start = k % self.n  # hoặc: int(self.rng.integers(self.n))
                t = self._build_tour(start)
                L = tour_length(t, self.D)
                tours.append(t)
                lengths.append(L)

            # Cập nhật best toàn cục
            i_best = int(np.argmin(lengths))
            if lengths[i_best] < self.best_length:
                self.best_length = float(lengths[i_best])
                self.best_tour = tours[i_best]

            # Cập nhật pheromone
            self._update_pheromones(tours, lengths)

            # Lưu lịch sử best-so-far
            history.append(self.best_length)

            if on_iter:
                # Chụp snapshot tau mỗi k vòng để tiết kiệm RAM
                extras = {}
                if it % 2 == 0:  # hoặc %5 tuỳ bạn
                    extras["tau"] = self.tau.copy()
                best_copy = self.best_tour[:] if self.best_tour else None
                on_iter(it, self.best_length, best_copy, extras)
            

        assert self.best_tour is not None
        return self.best_tour, self.best_length, history


@dataclass
class SensitivityEntry:
    """Aggregate statistics for a single parameter value."""

    param: str
    value: Any
    best_lengths: List[float]
    mean_length: float
    std_length: float


def analyze_aco_sensitivity(
    D: np.ndarray,
    param_grid: Dict[str, List[Any]],
    base_cfg: Optional[ACOConfig] = None,
    runs_per_value: int = 3,
    vary_seed: bool = True,
) -> Dict[str, List[SensitivityEntry]]:
    """
    Chạy ACO nhiều lần để phân tích độ nhạy với từng tham số.

    Args:
        D: Ma trận khoảng cách.
        param_grid: Dict của dạng {'rho': [0.3, 0.5, ...], 'n_iterations': [200, 400]}.
        base_cfg: Cấu hình gốc để lấy mặc định cho các tham số không thay đổi.
        runs_per_value: Số lần chạy lặp lại cho mỗi giá trị để lấy trung bình.
        vary_seed: Nếu True thì mỗi lần lặp sẽ tăng seed để tránh trùng kết quả.

    Returns:
        Dict[param_name] -> List[SensitivityEntry] chứa mean/std của best_length.
    """
    if not param_grid:
        raise ValueError("param_grid phải chứa ít nhất một tham số để phân tích.")
    if runs_per_value <= 0:
        raise ValueError("runs_per_value phải lớn hơn 0.")

    base_cfg = base_cfg or ACOConfig()
    results: Dict[str, List[SensitivityEntry]] = {}

    for param_name, values in param_grid.items():
        if not values:
            continue
        entries: List[SensitivityEntry] = []
        for value in values:
            best_lengths: List[float] = []
            for run_idx in range(runs_per_value):
                cfg = replace(base_cfg)
                if not hasattr(cfg, param_name):
                    raise AttributeError(f"ACOConfig không có tham số '{param_name}'.")
                setattr(cfg, param_name, value)
                if vary_seed and cfg.seed is not None:
                    cfg.seed = cfg.seed + run_idx
                solver = AntColonyTSP(D, cfg)
                _, best_len, _ = solver.run()
                best_lengths.append(best_len)
            mean_length = float(np.mean(best_lengths))
            std_length = float(np.std(best_lengths)) if len(best_lengths) > 1 else 0.0
            entries.append(
                SensitivityEntry(
                    param=param_name,
                    value=value,
                    best_lengths=best_lengths,
                    mean_length=mean_length,
                    std_length=std_length,
                )
            )
        results[param_name] = entries

    return results
