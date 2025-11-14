import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtWidgets import QButtonGroup

from algorithms.aco_tsp import AntColonyTSP, ACOConfig
from algorithms.ga_tsp import ga_tsp
from algorithms.hc_tsp import hill_climbing_tsp
from algorithms.sa_tsp import simulated_annealing_tsp
from problems.tsp import read_weight_matrix


class MplCanvas(FigureCanvas):
    """Thin wrapper to embed Matplotlib inside PyQt widgets."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


@dataclass
class ParamField:
    name: str
    label: str
    default: str
    cast: Callable[[str], object]
    optional: bool = False


class TSPVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trực quan hóa TSP - ACO / GA / SA / HC")
        self.resize(1200, 800)

        self.distance_matrix: Optional[np.ndarray] = None
        self.node_positions: Optional[np.ndarray] = None
        self.best_tour: Optional[List[int]] = None
        self.best_length: Optional[float] = None
        self.best_iteration: Optional[int] = None
        self.history: Optional[List[float]] = None
        self.iter_records: List[Dict[str, Any]] = []
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(600)
        self.playback_timer.timeout.connect(self._advance_playback)
        self.playback_index = 0
        self.summary_labels: Dict[str, Dict[str, QLabel]] = {}
        self.algorithm_results: Dict[str, Dict[str, Optional[Any]]] = {}
        self.algorithm_history: Dict[str, List[float]] = {}

        self.param_config: Dict[str, List[ParamField]] = {
            "ACO": [
                ParamField("n_ants", "Số kiến (bỏ trống = auto)", "", int, True),
                ParamField("n_iterations", "Số vòng lặp", "200", int),
                ParamField("alpha", "Alpha (pheromone)", "1.0", float),
                ParamField("beta", "Beta (heuristic)", "5.0", float),
                ParamField("rho", "Hệ số bay hơi (rho)", "0.45", float),
                ParamField("q", "Hệ số lắng đọng (q)", "75.0", float),
                ParamField("elitist_weight", "Elitist weight", "0.5", float),
                ParamField("seed", "Seed", "42", int),
            ],
            "GA": [
                ParamField("pop_size", "Kích thước quần thể", "260", int),
                ParamField("n_gen", "Số thế hệ", "800", int),
                ParamField("elite_ratio", "Tỉ lệ elitism", "0.08", float),
                ParamField("crossover_rate", "Xác suất lai ghép", "0.92", float),
                ParamField("mutation_rate", "Xác suất đột biến", "0.15", float),
                ParamField("tournament_k", "K của giải đấu", "3", int),
                ParamField("seed", "Seed", "123", int),
            ],
            "SA": [
                ParamField("n_iterations", "Số vòng lặp", "3000", int),
                ParamField("alpha", "Alpha làm nguội", "0.998", float),
                ParamField("steps_per_T", "Bước mỗi nhiệt độ", "30", int),
                ParamField("T0", "Nhiệt độ ban đầu (bỏ trống = auto)", "", float, True),
                ParamField("seed", "Seed", "123", int),
            ],
            "HC": [
                ParamField("n_iterations", "Số vòng lặp", "4000", int),
                ParamField("seed", "Seed", "42", int),
            ],
        }

        self.param_inputs: Dict[str, Dict[str, QLineEdit]] = {}
        self.aco_param_fields: Dict[str, ParamField] = {
            field.name: field for field in self.param_config["ACO"]
        }

        self._setup_ui()
        self._reset_summary()
        self._load_initial_matrix()
        self._ensure_params_editable()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        controls = QVBoxLayout()
        controls.setAlignment(Qt.AlignTop)

        # --- Data group
        data_group = QGroupBox("1. Dữ liệu TSP")
        data_layout = QGridLayout()
        self.file_input = QLineEdit("Discrete_Swarm_Optimization/data/weights_25.csv")
        self.load_button = QPushButton("Nạp ma trận")
        self.load_button.clicked.connect(self._load_matrix_from_input)
        self.matrix_info_label = QLabel("Chưa tải")
        data_layout.addWidget(QLabel("Đường dẫn:"), 0, 0)
        data_layout.addWidget(self.file_input, 0, 1)
        data_layout.addWidget(self.load_button, 1, 0, 1, 2)
        data_layout.addWidget(self.matrix_info_label, 2, 0, 1, 2)
        data_group.setLayout(data_layout)
        controls.addWidget(data_group)

        # --- Algorithm group
        algo_group = QGroupBox("2. Thuật toán và tham số")
        algo_layout = QVBoxLayout()
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(self.param_config.keys())
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        algo_layout.addWidget(QLabel("Thuật toán:"))
        algo_layout.addWidget(self.algorithm_combo)

        self.param_container = QTabWidget()
        self.param_container.setTabBarAutoHide(True)
        for name, fields in self.param_config.items():
            tab = QWidget()
            grid = QGridLayout()
            inputs = {}
            for row, field in enumerate(fields):
                edit = QLineEdit(field.default)
                edit.setReadOnly(False)
                edit.setEnabled(True)
                edit.setFocusPolicy(Qt.StrongFocus)
                inputs[field.name] = edit
                grid.addWidget(QLabel(field.label), row, 0)
                grid.addWidget(edit, row, 1)
            tab.setLayout(grid)
            self.param_inputs[name] = inputs
            self.param_container.addTab(tab, name)
        algo_layout.addWidget(self.param_container)
        algo_group.setLayout(algo_layout)
        controls.addWidget(algo_group)

        # --- Run and result group
        button_row = QHBoxLayout()
        self.run_button = QPushButton("Chạy thuật toán")
        self.run_button.clicked.connect(self._run_algorithm)
        button_row.addWidget(self.run_button)
        self.compare_button = QPushButton("Chạy tất cả (so sánh)")
        self.compare_button.clicked.connect(self._run_all_algorithms)
        button_row.addWidget(self.compare_button)
        self.reset_params_button = QPushButton("Reset tham số")
        self.reset_params_button.clicked.connect(self._reset_params)
        button_row.addWidget(self.reset_params_button)
        controls.addLayout(button_row)

        result_group = QGroupBox("3. Kết quả")
        result_layout = QVBoxLayout()
        self.best_length_label = QLabel("best_length: N/A")
        self.best_iter_label = QLabel("iteration(best): N/A")
        self.tour_label = QLabel("best_tour: N/A")
        self.tour_label.setWordWrap(True)
        result_layout.addWidget(self.best_length_label)
        result_layout.addWidget(self.best_iter_label)
        result_layout.addWidget(self.tour_label)
        result_group.setLayout(result_layout)
        controls.addWidget(result_group)

        controls.addStretch()
        main_layout.addLayout(controls, 1)

        # --- Visualization tabs
        self.tabs = QTabWidget()
        self.tab_tour = QWidget()
        self.tab_history = QWidget()
        self.tab_compare = QWidget()
        self.tabs.addTab(self.tab_tour, "Lộ trình")
        self.tabs.addTab(self.tab_history, "Đồ thị hội tụ")
        self.tabs.addTab(self.tab_compare, "So sánh")
        main_layout.addWidget(self.tabs, 2)

        tour_layout = QVBoxLayout(self.tab_tour)
        self.canvas_tour = MplCanvas(self.tab_tour, width=5, height=5)
        tour_layout.addWidget(self.canvas_tour)
        playback_controls = QHBoxLayout()
        self.playback_label = QLabel("Playback: N/A")
        self.skip_button = QPushButton("Bỏ qua")
        self.skip_button.setEnabled(False)
        self.skip_button.clicked.connect(self._skip_playback)
        playback_controls.addWidget(self.playback_label)
        playback_controls.addStretch()
        playback_controls.addWidget(self.skip_button)
        tour_layout.addLayout(playback_controls)

        history_layout = QVBoxLayout(self.tab_history)
        self.canvas_history = MplCanvas(self.tab_history, width=5, height=5)
        history_layout.addWidget(self.canvas_history)

        outer_compare_layout = QVBoxLayout(self.tab_compare)
        scroll_area = QScrollArea(self.tab_compare)
        scroll_area.setWidgetResizable(True)
        compare_container = QWidget()
        compare_layout = QVBoxLayout(compare_container)

        summary_group = QGroupBox("Tổng quan 4 thuật toán")
        summary_layout = QGridLayout()
        summary_layout.addWidget(QLabel("Thuật toán"), 0, 0)
        summary_layout.addWidget(QLabel("best_length"), 0, 1)
        summary_layout.addWidget(QLabel("best_tour"), 0, 2)

        for row, algo in enumerate(self.param_config.keys(), start=1):
            summary_layout.addWidget(QLabel(algo), row, 0)
            length_label = QLabel("N/A")
            tour_label = QLabel("N/A")
            tour_label.setWordWrap(True)
            summary_layout.addWidget(length_label, row, 1)
            summary_layout.addWidget(tour_label, row, 2)
            self.summary_labels[algo] = {
                "length": length_label,
                "tour": tour_label,
            }

        summary_group.setLayout(summary_layout)
        compare_layout.addWidget(summary_group)

        history_group = QGroupBox("Đồ thị hội tụ (4 thuật toán)")
        history_group_layout = QVBoxLayout()
        self.canvas_compare_history = MplCanvas(history_group, width=5, height=4)
        history_group_layout.addWidget(self.canvas_compare_history)
        history_group.setLayout(history_group_layout)
        compare_layout.addWidget(history_group)

        sensitivity_group = QGroupBox("Phân tích độ nhạy ACO (đa seed)")
        sensitivity_layout = QGridLayout()
        sensitivity_layout.addWidget(QLabel("Tham số"), 0, 0)
        param_buttons_layout = QHBoxLayout()
        self.sensitivity_param_group = QButtonGroup(self)
        self.sensitivity_param_buttons: Dict[str, QRadioButton] = {}
        for idx, name in enumerate(["alpha", "beta", "rho", "q", "elitist_weight"]):
            btn = QRadioButton(name)
            if idx == 0:
                btn.setChecked(True)
            self.sensitivity_param_group.addButton(btn)
            param_buttons_layout.addWidget(btn)
            self.sensitivity_param_buttons[name] = btn
        sensitivity_layout.addLayout(param_buttons_layout, 0, 1)

        sensitivity_layout.addWidget(QLabel("Giá trị (phân tách ,)"), 1, 0)
        self.sensitivity_values_input = QLineEdit("0.3, 0.5, 0.7")
        self._ensure_line_edit_editable(self.sensitivity_values_input)
        sensitivity_layout.addWidget(self.sensitivity_values_input, 1, 1)

        sensitivity_layout.addWidget(QLabel("Seeds (5 giá trị)"), 2, 0)
        self.sensitivity_seeds_input = QLineEdit("42, 99, 123, 2024, 4096")
        self._ensure_line_edit_editable(self.sensitivity_seeds_input)
        sensitivity_layout.addWidget(self.sensitivity_seeds_input, 2, 1)

        sensitivity_layout.addWidget(QLabel("ε (%)"), 3, 0)
        self.sensitivity_eps_input = QLineEdit("1")
        self._ensure_line_edit_editable(self.sensitivity_eps_input)
        sensitivity_layout.addWidget(self.sensitivity_eps_input, 3, 1)

        self.sensitivity_run_button = QPushButton("Phân tích ACO (đa seed)")
        self.sensitivity_run_button.clicked.connect(self._run_aco_sensitivity)
        sensitivity_layout.addWidget(self.sensitivity_run_button, 4, 0, 1, 2)

        self.sensitivity_status_label = QLabel("Chưa chạy phân tích.")
        sensitivity_layout.addWidget(self.sensitivity_status_label, 5, 0, 1, 2)

        self.sensitivity_table = QTableWidget(0, 5)
        self.sensitivity_table.setHorizontalHeaderLabels(
            [
                "Giá trị",
                "Len (mean ± std)",
                "Δ% vs baseline",
                "Iters-to-ε (mean ± std)",
                "Stability (std)",
            ]
        )
        sensitivity_layout.addWidget(self.sensitivity_table, 6, 0, 1, 2)
        sensitivity_group.setLayout(sensitivity_layout)
        compare_layout.addWidget(sensitivity_group)
        compare_layout.addStretch()

        scroll_area.setWidget(compare_container)
        outer_compare_layout.addWidget(scroll_area)

        self._on_algorithm_changed()

    def _load_initial_matrix(self) -> None:
        try:
            self._load_matrix(self.file_input.text())
        except Exception as exc:  # pragma: no cover - only hits when file missing
            self._show_error(f"Không thể tải ma trận mặc định: {exc}")

    def _load_matrix_from_input(self) -> None:
        path = self.file_input.text().strip()
        if not path:
            self._show_error("Vui lòng nhập đường dẫn ma trận.")
            return
        try:
            self._load_matrix(path)
        except Exception as exc:
            self._show_error(str(exc))

    def _load_matrix(self, path: str) -> None:
        self.distance_matrix = read_weight_matrix(path)
        n = self.distance_matrix.shape[0]
        self.node_positions = self._create_node_layout(n)
        self.matrix_info_label.setText(f"Ma trận: {n} x {n}")
        self.best_tour = None
        self.best_length = None
        self.best_iteration = None
        self.history = None
        self.best_length_label.setText("best_length: N/A")
        self.best_iter_label.setText("iteration(best): N/A")
        self.tour_label.setText("best_tour: N/A")
        self._reset_summary()
        self._ensure_params_editable()
        self._clear_sensitivity_results()
        self._clear_plots()
        self._reset_playback_state()

    def _on_algorithm_changed(self) -> None:
        idx = self.algorithm_combo.currentIndex()
        self.param_container.setCurrentIndex(idx)

    def _reset_params(self) -> None:
        algo = self.algorithm_combo.currentText()
        for field in self.param_config[algo]:
            self.param_inputs[algo][field.name].setText(field.default)

    def _run_algorithm(self) -> None:
        if self.distance_matrix is None:
            self._show_error("Chưa có ma trận khoảng cách. Hãy nạp dữ liệu trước.")
            return
        self._reset_playback_state()
        algo = self.algorithm_combo.currentText()
        try:
            params = self._collect_params(algo)
            best_tour, best_len, history = self._execute_algorithm(algo, params)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime safety net
            self._show_error(f"Lỗi khi chạy thuật toán: {exc}")
            return

        self.best_tour = best_tour
        self.best_length = best_len
        self.history = history
        self.best_length_label.setText(f"best_length: {best_len:.2f}")
        self.best_iteration = self._find_iteration_of_best(history, best_len)
        if self.best_iteration is None:
            self.best_iter_label.setText("iteration(best): N/A")
        else:
            self.best_iter_label.setText(f"iteration(best): {self.best_iteration}")
        self.tour_label.setText(" -> ".join(map(str, best_tour)))
        self._update_algo_summary(algo, best_len, best_tour)
        self.algorithm_history[algo] = history[:] if history else []
        self._plot_compare_history()

        self._plot_tour(best_tour, algo)
        self._plot_history(history, algo)
        self._start_playback()

    def _run_all_algorithms(self) -> None:
        if self.distance_matrix is None:
            self._show_error("Chưa có ma trận khoảng cách. Hãy nạp dữ liệu trước.")
            return

        for algo in self.param_config.keys():
            try:
                params = self._collect_params(algo)
            except ValueError as exc:
                self._show_error(f"Lỗi tham số ({algo}): {exc}")
                return
            try:
                best_tour, best_len, history = self._execute_algorithm(
                    algo, params, capture_history=False
                )
            except Exception as exc:  # pragma: no cover - runtime safety net
                self._show_error(f"Lỗi khi chạy {algo}: {exc}")
                return
            self._update_algo_summary(algo, best_len, best_tour)
            self.algorithm_history[algo] = history[:] if history else []

        self._plot_compare_history()

    def _collect_params(self, algo: Optional[str] = None) -> Dict[str, object]:
        if algo is None:
            algo = self.algorithm_combo.currentText()
        params = {}
        for field in self.param_config[algo]:
            text = self.param_inputs[algo][field.name].text().strip()
            if not text:
                if field.optional:
                    params[field.name] = None
                    continue
                raise ValueError(f"Tham số '{field.label}' không được để trống.")
            try:
                params[field.name] = field.cast(text)
            except ValueError as exc:
                raise ValueError(
                    f"Không thể chuyển '{field.label}' sang {field.cast.__name__}: {text}"
                ) from exc
        return params

    def _execute_algorithm(
        self, algo: str, params: Dict[str, object], capture_history: bool = True
    ):
        D = self.distance_matrix
        if D is None:
            raise ValueError("Ma trận chưa được nạp.")

        if algo == "ACO":
            cfg = ACOConfig(
                n_ants=params["n_ants"],
                n_iterations=params["n_iterations"],
                alpha=params["alpha"],
                beta=params["beta"],
                rho=params["rho"],
                q=params["q"],
                elitist_weight=params["elitist_weight"],
                seed=params["seed"],
            )
            solver = AntColonyTSP(D, cfg)
            recorder = (
                self._make_recorder(algo, cfg.n_iterations) if capture_history else None
            )
            callback = (
                (lambda it, best_len, best_tour, extras: recorder(it, best_len, best_tour))
                if recorder
                else None
            )
            return solver.run(on_iter=callback)

        if algo == "GA":
            recorder = (
                self._make_recorder(algo, params["n_gen"]) if capture_history else None
            )
            return ga_tsp(
                D,
                pop_size=params["pop_size"],
                n_gen=params["n_gen"],
                elite_ratio=params["elite_ratio"],
                crossover_rate=params["crossover_rate"],
                mutation_rate=params["mutation_rate"],
                tournament_k=params["tournament_k"],
                seed=params["seed"],
                on_iter=(
                    (lambda gen, best_len, best_tour: recorder(gen, best_len, best_tour))
                    if recorder
                    else None
                ),
            )

        if algo == "SA":
            recorder = (
                self._make_recorder(algo, params["n_iterations"])
                if capture_history
                else None
            )
            return simulated_annealing_tsp(
                D,
                n_iterations=params["n_iterations"],
                T0=params["T0"],
                alpha=params["alpha"],
                seed=params["seed"],
                steps_per_T=params["steps_per_T"],
                on_iter=(
                    (lambda it, best_len, best_tour: recorder(it, best_len, best_tour))
                    if recorder
                    else None
                ),
            )

        if algo == "HC":
            recorder = (
                self._make_recorder(algo, params["n_iterations"])
                if capture_history
                else None
            )
            return hill_climbing_tsp(
                D,
                n_iterations=params["n_iterations"],
                seed=params["seed"],
                on_iter=(
                    (lambda it, best_len, best_tour: recorder(it, best_len, best_tour))
                    if recorder
                    else None
                ),
            )

        raise ValueError(f"Thuật toán '{algo}' chưa được hỗ trợ.")

    def _plot_tour(self, tour: List[int], algo: str) -> None:
        title = f"Lộ trình tốt nhất ({algo})"
        self._draw_tour(self.canvas_tour.axes, tour, title)
        self.canvas_tour.draw()

    def _plot_history(self, history: List[float], algo: str) -> None:
        ax = self.canvas_history.axes
        ax.clear()
        if not history:
            ax.set_title("Chưa có lịch sử hội tụ.")
        else:
            ax.plot(history, color="#03045e")
            ax.set_xlabel("Iteration / Generation")
            ax.set_ylabel("Best tour length")
            ax.set_title(f"Hội tụ - {algo}")
            ax.grid(True, alpha=0.3)
        self.canvas_history.draw()

    def _record_iteration(self, iteration: int, length: float, tour: List[int]) -> None:
        snapshot = {
            "iteration": int(iteration),
            "length": float(length),
            "tour": tour[:],
        }
        self.iter_records.append(snapshot)

    def _make_recorder(self, algo: str, total_iters: int):
        stride = self._compute_capture_stride(algo, total_iters)
        last_iter = -1

        def recorder(iter_idx: int, best_len: float, best_tour: Optional[List[int]]):
            nonlocal last_iter
            if best_tour is None:
                return
            is_last = iter_idx >= total_iters - 1
            if (iter_idx % stride == 0 or is_last) and iter_idx != last_iter:
                self._record_iteration(iter_idx, best_len, best_tour)
                last_iter = iter_idx

        return recorder

    @staticmethod
    def _compute_capture_stride(algo: str, total_iters: int) -> int:
        stride_map = {
            "ACO": 10,
            "GA": 10,
            "SA": 50,
            "HC": 50,
        }
        stride = stride_map.get(algo.upper(), 10)
        return max(1, stride)

    def _start_playback(self) -> None:
        if not self.iter_records:
            self._show_final_tour()
            return
        self.playback_index = 0
        if hasattr(self, "skip_button"):
            self.skip_button.setEnabled(True)
        self.playback_timer.start()
        self._advance_playback()

    def _advance_playback(self) -> None:
        if not self.iter_records:
            self._stop_playback()
            return
        if self.playback_index >= len(self.iter_records):
            self._show_final_tour("(xong)")
            return
        record = self.iter_records[self.playback_index]
        title = f"Iteration {record['iteration']} - Best {record['length']:.2f}"
        self._draw_tour(self.canvas_tour.axes, record["tour"], title)
        self.canvas_tour.draw()
        self.playback_label.setText(title)
        self.playback_index += 1

    def _skip_playback(self) -> None:
        if not self.iter_records:
            return
        self.playback_index = len(self.iter_records)
        self._show_final_tour("(đã bỏ qua)")

    def _stop_playback(self) -> None:
        if self.playback_timer.isActive():
            self.playback_timer.stop()
        if hasattr(self, "skip_button"):
            self.skip_button.setEnabled(False)

    def _show_final_tour(self, suffix: str = "") -> None:
        self._stop_playback()
        if self.iter_records:
            final = self.iter_records[-1]
            title = f"Iteration {final['iteration']} - Best {final['length']:.2f}"
            self._draw_tour(self.canvas_tour.axes, final["tour"], title)
            self.canvas_tour.draw()
            self.playback_label.setText(f"{title} {suffix}".strip())
        elif self.best_tour is not None:
            self._draw_tour(self.canvas_tour.axes, self.best_tour, "Lộ trình tối ưu")
            self.canvas_tour.draw()
            self.playback_label.setText("Lộ trình tối ưu")
        else:
            self.playback_label.setText("Playback: N/A")

    def _reset_playback_state(self) -> None:
        self._stop_playback()
        self.iter_records = []
        self.playback_index = 0
        if hasattr(self, "playback_label"):
            self.playback_label.setText("Playback: N/A")
        if hasattr(self, "skip_button"):
            self.skip_button.setEnabled(False)

    def _draw_tour(self, ax, tour: Optional[List[int]], title: str) -> None:
        ax.clear()
        if tour is None or self.node_positions is None:
            ax.set_title("Chưa có dữ liệu để vẽ.")
            return
        coords = np.array([self.node_positions[i] for i in tour])
        all_nodes = self.node_positions
        ax.plot(coords[:, 0], coords[:, 1], color="#1d3557", lw=1.2, alpha=0.6)
        ax.scatter(all_nodes[:, 0], all_nodes[:, 1], color="#f3722c", s=26, zorder=5)
        # Arrowheads to emphasize direction
        for i in range(len(coords) - 1):
            ax.annotate(
                "",
                xy=(coords[i + 1, 0], coords[i + 1, 1]),
                xytext=(coords[i, 0], coords[i, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#0077b6",
                    lw=1.0,
                    shrinkA=6,
                    shrinkB=6,
                    mutation_scale=12,
                    alpha=0.8,
                ),
            )
        start_point = self.node_positions[tour[0]]
        ax.scatter(
            start_point[0],
            start_point[1],
            color="#2a9d8f",
            s=160,
            edgecolors="white",
            linewidths=1.5,
            zorder=7,
        )
        for idx, (x, y) in enumerate(all_nodes):
            ax.text(
                x,
                y,
                str(idx),
                color="#1d3557",
                fontsize=8,
                ha="center",
                va="center",
                zorder=6,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.axis("off")

    @staticmethod
    def _create_node_layout(n: int) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 1.0
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.column_stack([x, y])

    def _clear_plots(self) -> None:
        self.canvas_tour.axes.clear()
        self.canvas_history.axes.clear()
        self.canvas_tour.draw()
        self.canvas_history.draw()

    @staticmethod
    def _find_iteration_of_best(history: Optional[List[float]], best_len: Optional[float]) -> Optional[int]:
        if not history or best_len is None:
            return None
        for idx, value in enumerate(history):
            if math.isclose(value, best_len, rel_tol=1e-9, abs_tol=1e-9):
                return idx + 1  # 1-based iteration/generation count
        return len(history)

    @staticmethod
    def _compute_speed_to_threshold(history: Optional[List[float]], threshold: float) -> int:
        if not history:
            return 0
        for idx, value in enumerate(history, start=1):
            if value <= threshold:
                return idx
        return len(history)

    def _reset_summary(self) -> None:
        self.algorithm_results = {
            algo: {"length": None, "tour": None} for algo in self.param_config.keys()
        }
        self.algorithm_history = {algo: [] for algo in self.param_config.keys()}
        for labels in self.summary_labels.values():
            labels["length"].setText("N/A")
            labels["tour"].setText("N/A")
        self._plot_compare_history()

    def _ensure_params_editable(self) -> None:
        for inputs in self.param_inputs.values():
            for edit in inputs.values():
                self._ensure_line_edit_editable(edit)

    @staticmethod
    def _ensure_line_edit_editable(edit: QLineEdit) -> None:
        edit.setReadOnly(False)
        edit.setEnabled(True)
        edit.setFocusPolicy(Qt.StrongFocus)

    def _update_algo_summary(self, algo: str, length: float, tour: List[int]) -> None:
        if algo not in self.algorithm_results:
            return
        self.algorithm_results[algo] = {"length": length, "tour": tour[:]}
        labels = self.summary_labels.get(algo)
        if not labels:
            return
        labels["length"].setText(f"{length:.2f}")
        labels["tour"].setText(" -> ".join(map(str, tour)))

    def _run_aco_sensitivity(self) -> None:
        if self.distance_matrix is None:
            self._show_error("Chưa có ma trận khoảng cách. Hãy nạp dữ liệu trước.")
            return
        param_name = self._get_selected_sensitivity_param()
        if not param_name:
            self._show_error("Chưa chọn tham số.")
            return
        try:
            values = self._parse_sensitivity_values(param_name)
            seeds = self._parse_seed_list(self.sensitivity_seeds_input.text())
            if len(seeds) < 1:
                raise ValueError("Cần ít nhất 1 seed để phân tích.")
            epsilon_pct = float(self.sensitivity_eps_input.text().strip() or "0")
            if epsilon_pct < 0:
                raise ValueError("ε phải không âm.")
            epsilon_ratio = epsilon_pct / 100.0
            base_params = self._collect_params("ACO")
        except ValueError as exc:
            self._show_error(str(exc))
            return

        def make_cfg(seed_value: int, override_value: Any) -> ACOConfig:
            cfg = ACOConfig(
                n_ants=base_params["n_ants"],
                n_iterations=base_params["n_iterations"],
                alpha=base_params["alpha"],
                beta=base_params["beta"],
                rho=base_params["rho"],
                q=base_params["q"],
                elitist_weight=base_params["elitist_weight"],
                seed=seed_value,
            )
            setattr(cfg, param_name, override_value)
            return cfg

        baseline_value = base_params.get(param_name)
        baseline_runs: List[Dict[str, Any]] = []
        global_best = float("inf")
        for seed in seeds:
            cfg = make_cfg(seed, baseline_value)
            solver = AntColonyTSP(self.distance_matrix, cfg)
            _, best_len, history = solver.run()
            baseline_runs.append({"best_len": best_len, "history": history})
            if best_len < global_best:
                global_best = best_len

        entries: List[Dict[str, Any]] = []
        for value in values:
            runs: List[Dict[str, Any]] = []
            for seed in seeds:
                cfg = make_cfg(seed, value)
                solver = AntColonyTSP(self.distance_matrix, cfg)
                _, best_len, history = solver.run()
                runs.append({"best_len": best_len, "history": history})
                if best_len < global_best:
                    global_best = best_len
            entries.append({"value": value, "runs": runs})

        if not entries:
            self._show_error("Không có giá trị nào để chạy.")
            return

        eps_threshold = global_best * (1.0 + epsilon_ratio)
        baseline_mean = float(np.mean([r["best_len"] for r in baseline_runs]))
        table_rows: List[List[str]] = []
        for entry in entries:
            best_lengths = [run["best_len"] for run in entry["runs"]]
            histories = [run["history"] for run in entry["runs"]]
            mean_len = float(np.mean(best_lengths))
            std_len = float(np.std(best_lengths)) if len(best_lengths) > 1 else 0.0
            speeds = [
                self._compute_speed_to_threshold(hist, eps_threshold)
                for hist in histories
            ]
            speed_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0
            mean_speed = float(np.mean(speeds)) if speeds else float("nan")
            delta_pct = (
                ((mean_len - baseline_mean) / baseline_mean * 100.0)
                if baseline_mean > 0
                else float("nan")
            )
            table_rows.append(
                [
                    str(entry["value"]),
                    f"{mean_len:.2f} ± {std_len:.2f}",
                    f"{delta_pct:.2f}%",
                    f"{mean_speed:.1f} ± {speed_std:.1f}",
                    f"{std_len:.2f}",
                ]
            )

        self._update_sensitivity_table(table_rows)
        self.sensitivity_status_label.setText(
            f"Đã chạy {len(entries)} giá trị, {len(seeds)} seed, ε={epsilon_pct:.2f}%, baseline={baseline_mean:.2f}."
        )

    def _parse_sensitivity_values(self, param_name: str) -> List[Any]:
        text = self.sensitivity_values_input.text().strip()
        if not text:
            raise ValueError("Vui lòng nhập danh sách giá trị.")
        field = self.aco_param_fields.get(param_name)
        if field is None:
            raise ValueError(f"Tham số '{param_name}' không hợp lệ.")
        caster = field.cast
        values: List[Any] = []
        for chunk in text.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if field.optional and chunk.lower() in {"auto", "none"}:
                values.append(None)
                continue
            try:
                values.append(caster(chunk))
            except ValueError as exc:
                raise ValueError(
                    f"Không thể chuyển '{chunk}' sang {caster.__name__}"
                ) from exc
        if not values:
            raise ValueError("Danh sách giá trị trống.")
        return values

    @staticmethod
    def _parse_seed_list(text: str) -> List[int]:
        cleaned = text.replace(";", ",").split(",")
        seeds: List[int] = []
        for item in cleaned:
            item = item.strip()
            if not item:
                continue
            seeds.append(int(item))
        return seeds

    def _get_selected_sensitivity_param(self) -> Optional[str]:
        for name, btn in self.sensitivity_param_buttons.items():
            if btn.isChecked():
                return name
        return None

    def _update_sensitivity_table(self, rows: List[List[str]]) -> None:
        table = self.sensitivity_table
        table.setRowCount(len(rows))
        for r, row_data in enumerate(rows):
            for c, value in enumerate(row_data):
                table.setItem(r, c, QTableWidgetItem(value))
        table.resizeColumnsToContents()

    def _clear_sensitivity_results(self) -> None:
        if hasattr(self, "sensitivity_table"):
            self.sensitivity_table.setRowCount(0)
        if hasattr(self, "sensitivity_status_label"):
            self.sensitivity_status_label.setText("Chưa chạy phân tích.")

    def _plot_compare_history(self) -> None:
        if not hasattr(self, "canvas_compare_history"):
            return
        canvas = self.canvas_compare_history
        ax = canvas.axes
        ax.clear()
        has_data = False
        for algo, history in self.algorithm_history.items():
            if history:
                ax.plot(history, label=algo)
                has_data = True
        if has_data:
            ax.set_title("So sánh hội tụ")
            ax.set_xlabel("Iteration / Generation")
            ax.set_ylabel("Best tour length")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title("Chưa có dữ liệu so sánh.")
        canvas.fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.18)
        canvas.draw()

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Lỗi", message)


def main():
    app = QApplication(sys.argv)
    window = TSPVisualizer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
