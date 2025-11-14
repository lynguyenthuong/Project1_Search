import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QTabWidget, QGridLayout, QMessageBox, QGroupBox, QSpacerItem,
    QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from artificial_bee_colony import ArtificialBeeColony
from classic_algorithms import GeneticAlgorithm, HillClimbing, SimulatedAnnealing
from firefly_algorithm import FireflyAlgorithm
from particle_swarm_optimization import ParticleSwarmOptimization
from cuckoo_search import CuckooSearch
from fitness_function import (
    ackley_function, rastrigin_function, rosenbrock_function,
    sphere_function, get_function_bounds, get_global_min_pos
)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trực Quan Hóa Thuật Toán Tối Ưu")
        self.setGeometry(0, 0, 1200, 800)

        self.history_abc = None
        self.best_abc = None
        self.fitness_abc = None
        self.H_fit_abc = None

        self.history_ga = None
        self.best_ga = None
        self.fitness_ga = None
        self.H_fit_ga = None

        self.history_hc = None
        self.best_hc = None
        self.fitness_hc = None
        self.H_fit_hc = None

        self.history_sa = None
        self.best_sa = None
        self.fitness_sa = None
        self.H_fit_sa = None

        self.history_fa = None
        self.best_fa = None
        self.fitness_fa = None
        self.H_fit_fa = None

        self.history_pso = None
        self.best_pso = None
        self.fitness_pso = None
        self.H_fit_pso = None

        self.history_cs = None
        self.best_cs = None
        self.fitness_cs = None
        self.H_fit_cs = None

        self.current_animation = None

        self.algo_colors = {
            "ABC": "blue", "PSO": "cyan", "FA": "purple", "CS": "brown",
            "GA": "red", "SA": "orange", "HC": "green"
        }

        self._setup_ui()

    def _setup_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_panel_widget = QWidget()
        control_panel_layout = QVBoxLayout(control_panel_widget)
        control_panel_layout.setAlignment(Qt.AlignTop)
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel_widget)
        scroll_area.setWidgetResizable(True)

        func_group = QGroupBox("1. Chọn Hàm Fitness")
        func_layout = QGridLayout()

        self.function_map = {
            "Ackley Function": ackley_function,
            "Rastrigin Function": rastrigin_function,
            "Rosenbrock Function": rosenbrock_function,
            "Sphere Function": sphere_function,
        }
        self.func_combo = QComboBox()
        self.func_combo.addItems(self.function_map.keys())
        self.func_combo.setCurrentText("Ackley Function")
        self.func_combo.currentIndexChanged.connect(
            self._update_bounds_display)

        self.bounds_label = QLabel("Giới hạn: [-32.768, 32.768]")

        func_layout.addWidget(QLabel("Hàm Fitness:"), 0, 0)
        func_layout.addWidget(self.func_combo, 0, 1)
        func_layout.addWidget(self.bounds_label, 1, 0, 1, 2)
        func_group.setLayout(func_layout)
        control_panel_layout.addWidget(func_group)

        param_group = QGroupBox("2. Chế độ Chạy & Tham số Chung")
        param_layout = QGridLayout()

        self.run_mode_combo = QComboBox()
        self.run_mode_combo.addItems(
            ["Theo Vòng Lặp (Iteration)", "Theo Số Lần Gọi Hàm (NFE)"])
        self.run_mode_combo.currentIndexChanged.connect(
            self._update_param_visibility)

        self.max_iter_label = QLabel("Số vòng lặp (Max Iter):")
        self.max_iter_input = QLineEdit("50")

        self.max_nfe_label = QLabel("Số lần gọi hàm (Max NFE):")
        self.max_nfe_input = QLineEdit("5000")

        self.pop_size_label = QLabel("Kích thước quần thể (Pop Size):")
        self.pop_size_input = QLineEdit("40")

        param_layout.addWidget(QLabel("Chế độ chạy:"), 0, 0)
        param_layout.addWidget(self.run_mode_combo, 0, 1)

        param_layout.addWidget(self.max_iter_label, 1, 0)
        param_layout.addWidget(self.max_iter_input, 1, 1)

        param_layout.addWidget(self.max_nfe_label, 2, 0)
        param_layout.addWidget(self.max_nfe_input, 2, 1)

        param_layout.addWidget(self.pop_size_label, 3, 0)
        param_layout.addWidget(self.pop_size_input, 3, 1)

        param_group.setLayout(param_layout)
        control_panel_layout.addWidget(param_group)

        self._update_param_visibility()

        abc_group = QGroupBox("3. Artificial Bee Colony (ABC)")
        abc_layout = QGridLayout()
        self.run_abc_button = QPushButton("Chạy ABC")
        self.run_abc_button.clicked.connect(self._run_abc)
        self.abc_limit_input = QLineEdit("10")
        self.abc_fitness_label = QLabel("ABC Best Fitness: N/A")
        self.abc_solution_label = QLabel("ABC Best Solution: N/A")

        abc_layout.addWidget(self.run_abc_button, 0, 0, 1, 2)
        abc_layout.addWidget(QLabel("Giới hạn (Limit):"), 1, 0)
        abc_layout.addWidget(self.abc_limit_input, 1, 1)
        abc_layout.addWidget(self.abc_fitness_label, 2, 0, 1, 2)
        abc_layout.addWidget(self.abc_solution_label, 3, 0, 1, 2)
        abc_group.setLayout(abc_layout)
        control_panel_layout.addWidget(abc_group)

        pso_group = QGroupBox("4. Particle Swarm Optimization (PSO)")
        pso_layout = QGridLayout()
        self.run_pso_button = QPushButton("Chạy PSO")
        self.run_pso_button.clicked.connect(self._run_pso)
        self.pso_w_input = QLineEdit("0.7")
        self.pso_c1_input = QLineEdit("1.5")
        self.pso_c2_input = QLineEdit("1.5")
        self.pso_fitness_label = QLabel("PSO Best Fitness: N/A")
        self.pso_solution_label = QLabel("PSO Best Solution: N/A")

        pso_layout.addWidget(self.run_pso_button, 0, 0, 1, 2)
        pso_layout.addWidget(QLabel("Quán tính (w):"), 1, 0)
        pso_layout.addWidget(self.pso_w_input, 1, 1)
        pso_layout.addWidget(QLabel("Nhận thức (c1):"), 2, 0)
        pso_layout.addWidget(self.pso_c1_input, 2, 1)
        pso_layout.addWidget(QLabel("Xã hội (c2):"), 3, 0)
        pso_layout.addWidget(self.pso_c2_input, 3, 1)
        pso_layout.addWidget(self.pso_fitness_label, 4, 0, 1, 2)
        pso_layout.addWidget(self.pso_solution_label, 5, 0, 1, 2)
        pso_group.setLayout(pso_layout)
        control_panel_layout.addWidget(pso_group)

        fa_group = QGroupBox("5. Firefly Algorithm (FA)")
        fa_layout = QGridLayout()
        self.run_fa_button = QPushButton("Chạy FA")
        self.run_fa_button.clicked.connect(self._run_fa)
        self.fa_alpha_input = QLineEdit("0.2")
        self.fa_beta0_input = QLineEdit("1.0")
        self.fa_gamma_input = QLineEdit("1.0")
        self.fa_fitness_label = QLabel("FA Best Fitness: N/A")
        self.fa_solution_label = QLabel("FA Best Solution: N/A")

        fa_layout.addWidget(self.run_fa_button, 0, 0, 1, 2)
        fa_layout.addWidget(QLabel("Hệ số ngẫu nhiên (Alpha):"), 1, 0)
        fa_layout.addWidget(self.fa_alpha_input, 1, 1)
        fa_layout.addWidget(QLabel("Hấp dẫn ban đầu (Beta0):"), 2, 0)
        fa_layout.addWidget(self.fa_beta0_input, 2, 1)
        fa_layout.addWidget(QLabel("Suy giảm hấp dẫn (Gamma):"), 3, 0)
        fa_layout.addWidget(self.fa_gamma_input, 3, 1)
        fa_layout.addWidget(self.fa_fitness_label, 4, 0, 1, 2)
        fa_layout.addWidget(self.fa_solution_label, 5, 0, 1, 2)
        fa_group.setLayout(fa_layout)
        control_panel_layout.addWidget(fa_group)

        cs_group = QGroupBox("6. Cuckoo Search (CS)")
        cs_layout = QGridLayout()
        self.run_cs_button = QPushButton("Chạy CS")
        self.run_cs_button.clicked.connect(self._run_cs)
        self.cs_pa_input = QLineEdit("0.25")
        self.cs_alpha_input = QLineEdit("0.5")
        self.cs_fitness_label = QLabel("CS Best Fitness: N/A")
        self.cs_solution_label = QLabel("CS Best Solution: N/A")

        cs_layout.addWidget(self.run_cs_button, 0, 0, 1, 2)
        cs_layout.addWidget(QLabel("Xác suất phát hiện (pa):"), 1, 0)
        cs_layout.addWidget(self.cs_pa_input, 1, 1)
        cs_layout.addWidget(QLabel("Hệ số bước (alpha):"), 2, 0)
        cs_layout.addWidget(self.cs_alpha_input, 2, 1)
        cs_layout.addWidget(self.cs_fitness_label, 3, 0, 1, 2)
        cs_layout.addWidget(self.cs_solution_label, 4, 0, 1, 2)
        cs_group.setLayout(cs_layout)
        control_panel_layout.addWidget(cs_group)

        ga_group = QGroupBox("7. Genetic Algorithm (GA)")
        ga_layout = QGridLayout()
        self.run_ga_button = QPushButton("Chạy GA")
        self.run_ga_button.clicked.connect(self._run_ga)
        self.ga_crossover_input = QLineEdit("0.8")
        self.ga_mutation_input = QLineEdit("0.05")
        self.ga_fitness_label = QLabel("GA Best Fitness: N/A")
        self.ga_solution_label = QLabel("GA Best Solution: N/A")

        ga_layout.addWidget(self.run_ga_button, 0, 0, 1, 2)
        ga_layout.addWidget(QLabel("Tỷ lệ lai ghép (Crossover):"), 1, 0)
        ga_layout.addWidget(self.ga_crossover_input, 1, 1)
        ga_layout.addWidget(QLabel("Tỷ lệ đột biến (Mutation):"), 2, 0)
        ga_layout.addWidget(self.ga_mutation_input, 2, 1)
        ga_layout.addWidget(self.ga_fitness_label, 3, 0, 1, 2)
        ga_layout.addWidget(self.ga_solution_label, 4, 0, 1, 2)
        ga_group.setLayout(ga_layout)
        control_panel_layout.addWidget(ga_group)

        sa_group = QGroupBox("8. Simulated Annealing (SA)")
        sa_layout = QGridLayout()
        self.run_sa_button = QPushButton("Chạy SA")
        self.run_sa_button.clicked.connect(self._run_sa)
        self.sa_temp_input = QLineEdit("10.0")
        self.sa_cooling_input = QLineEdit("0.95")
        self.sa_step_size_input = QLineEdit("0.1")
        self.sa_fitness_label = QLabel("SA Best Fitness: N/A")
        self.sa_solution_label = QLabel("SA Best Solution: N/A")

        sa_layout.addWidget(self.run_sa_button, 0, 0, 1, 2)
        sa_layout.addWidget(QLabel("Nhiệt độ ban đầu (Temp):"), 1, 0)
        sa_layout.addWidget(self.sa_temp_input, 1, 1)
        sa_layout.addWidget(QLabel("Tỷ lệ làm mát (Cooling):"), 2, 0)
        sa_layout.addWidget(self.sa_cooling_input, 2, 1)
        sa_layout.addWidget(QLabel("Kích thước bước (Step Size):"), 3, 0)
        sa_layout.addWidget(self.sa_step_size_input, 3, 1)
        sa_layout.addWidget(self.sa_fitness_label, 4, 0, 1, 2)
        sa_layout.addWidget(self.sa_solution_label, 5, 0, 1, 2)
        sa_group.setLayout(sa_layout)
        control_panel_layout.addWidget(sa_group)

        hc_group = QGroupBox("9. Hill Climbing (HC)")
        hc_layout = QGridLayout()
        self.run_hc_button = QPushButton("Chạy HC")
        self.run_hc_button.clicked.connect(self._run_hc)
        self.hc_step_size_input = QLineEdit("0.1")
        self.hc_fitness_label = QLabel("HC Best Fitness: N/A")
        self.hc_solution_label = QLabel("HC Best Solution: N/M")

        hc_layout.addWidget(self.run_hc_button, 0, 0, 1, 2)
        hc_layout.addWidget(QLabel("Kích thước bước (Step Size):"), 1, 0)
        hc_layout.addWidget(self.hc_step_size_input, 1, 1)
        hc_layout.addWidget(self.hc_fitness_label, 2, 0, 1, 2)
        hc_layout.addWidget(self.hc_solution_label, 3, 0, 1, 2)
        hc_group.setLayout(hc_layout)
        control_panel_layout.addWidget(hc_group)

        control_panel_layout.addSpacerItem(QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        main_layout.addWidget(scroll_area, 1)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, 3)

        self.tab_3d = QWidget()
        self.tab_convergence_iter = QWidget()
        self.tab_convergence_nfe = QWidget()
        self.tab_contour = QWidget()
        self.tab_animation_abc = QWidget()
        self.tab_animation_pso = QWidget()
        self.tab_animation_fa = QWidget()
        self.tab_animation_cs = QWidget()
        self.tab_animation_ga = QWidget()
        self.tab_animation_sa = QWidget()
        self.tab_animation_hc = QWidget()

        self.tab_widget.addTab(self.tab_3d, "3D")
        self.tab_widget.addTab(self.tab_convergence_iter, "Converge (Iter)")
        self.tab_widget.addTab(self.tab_convergence_nfe, "Converge (NFE)")
        self.tab_widget.addTab(self.tab_contour, "Contour")
        self.tab_widget.addTab(self.tab_animation_abc, "ABC")
        self.tab_widget.addTab(self.tab_animation_pso, "PSO")
        self.tab_widget.addTab(self.tab_animation_fa, "FA")
        self.tab_widget.addTab(self.tab_animation_cs, "CS")
        self.tab_widget.addTab(self.tab_animation_ga, "GA")
        self.tab_widget.addTab(self.tab_animation_sa, "SA")
        self.tab_widget.addTab(self.tab_animation_hc, "HC")

        self.canvas_3d = MplCanvas(self.tab_3d, width=5, height=5)

        self.canvas_convergence_iter = MplCanvas(
            self.tab_convergence_iter, width=5, height=5)
        self.canvas_convergence_nfe = MplCanvas(
            self.tab_convergence_nfe, width=5, height=5)

        self.canvas_contour = MplCanvas(self.tab_contour, width=5, height=5)
        self.canvas_animation_abc = MplCanvas(
            self.tab_animation_abc, width=5, height=5)
        self.canvas_animation_pso = MplCanvas(
            self.tab_animation_pso, width=5, height=5)
        self.canvas_animation_fa = MplCanvas(
            self.tab_animation_fa, width=5, height=5)
        self.canvas_animation_cs = MplCanvas(
            self.tab_animation_cs, width=5, height=5)
        self.canvas_animation_ga = MplCanvas(
            self.tab_animation_ga, width=5, height=5)
        self.canvas_animation_sa = MplCanvas(
            self.tab_animation_sa, width=5, height=5)
        self.canvas_animation_hc = MplCanvas(
            self.tab_animation_hc, width=5, height=5)

        layout_3d = QVBoxLayout(self.tab_3d)
        layout_3d.addWidget(self.canvas_3d)
        layout_convergence_iter = QVBoxLayout(self.tab_convergence_iter)
        layout_convergence_iter.addWidget(self.canvas_convergence_iter)
        layout_convergence_nfe = QVBoxLayout(self.tab_convergence_nfe)
        layout_convergence_nfe.addWidget(self.canvas_convergence_nfe)
        layout_contour = QVBoxLayout(self.tab_contour)
        layout_contour.addWidget(self.canvas_contour)
        layout_animation_abc = QVBoxLayout(self.tab_animation_abc)
        layout_animation_abc.addWidget(self.canvas_animation_abc)
        layout_animation_pso = QVBoxLayout(self.tab_animation_pso)
        layout_animation_pso.addWidget(self.canvas_animation_pso)
        layout_animation_fa = QVBoxLayout(self.tab_animation_fa)
        layout_animation_fa.addWidget(self.canvas_animation_fa)
        layout_animation_cs = QVBoxLayout(self.tab_animation_cs)
        layout_animation_cs.addWidget(self.canvas_animation_cs)
        layout_animation_ga = QVBoxLayout(self.tab_animation_ga)
        layout_animation_ga.addWidget(self.canvas_animation_ga)
        layout_animation_sa = QVBoxLayout(self.tab_animation_sa)
        layout_animation_sa.addWidget(self.canvas_animation_sa)
        layout_animation_hc = QVBoxLayout(self.tab_animation_hc)
        layout_animation_hc.addWidget(self.canvas_animation_hc)

        self.tab_widget.currentChanged.connect(self._stop_all_animations)

        self._update_bounds_display()
        self._plot_3d()

    def _update_param_visibility(self):

        mode = self.run_mode_combo.currentText()
        if "Iteration" in mode:
            self.max_iter_label.show()
            self.max_iter_input.show()
            self.max_nfe_label.hide()
            self.max_nfe_input.hide()
            self.pop_size_label.setText("Kích thước quần thể (Pop Size):")
        else:
            self.max_iter_label.hide()
            self.max_iter_input.hide()
            self.max_nfe_label.show()
            self.max_nfe_input.show()
            self.pop_size_label.setText("Kích thước quần thể (Pop Size):")

    def _stop_all_animations(self):

        if self.current_animation:
            self.current_animation.event_source.stop()
        self.current_animation = None

        current_index = self.tab_widget.currentIndex()

        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = self.current_bounds

        if current_index == 4 and self.history_abc is not None:
            self._plot_animation_2d(
                selected_func, l_bound, u_bound, self.history_abc, "ABC", self.canvas_animation_abc)
        elif current_index == 5 and self.history_pso is not None:
            self._plot_animation_2d(
                selected_func, l_bound, u_bound, self.history_pso, "PSO", self.canvas_animation_pso)
        elif current_index == 6 and self.history_fa is not None:
            self._plot_animation_2d(
                selected_func, l_bound, u_bound, self.history_fa, "FA", self.canvas_animation_fa)
        elif current_index == 7 and self.history_cs is not None:
            self._plot_animation_2d(
                selected_func, l_bound, u_bound, self.history_cs, "CS", self.canvas_animation_cs)
        elif current_index == 8 and self.history_ga is not None:
            self._plot_animation_2d(
                selected_func, l_bound, u_bound, self.history_ga, "GA", self.canvas_animation_ga)
        elif current_index == 9 and self.history_sa is not None:
            self._plot_animation_2d(selected_func, l_bound, u_bound, self.history_sa,
                                    "SA", self.canvas_animation_sa, single_point=True)
        elif current_index == 10 and self.history_hc is not None:
            self._plot_animation_2d(selected_func, l_bound, u_bound, self.history_hc,
                                    "HC", self.canvas_animation_hc, single_point=True)

    def _update_bounds_display(self):

        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = get_function_bounds(selected_func)
        self.current_bounds = (l_bound, u_bound)
        self.bounds_label.setText(f"Giới hạn: [{l_bound}, {u_bound}]")
        self._plot_3d()
        self._reset_results_and_plots()

    def _get_parameters(self, validate=True):

        try:
            run_mode_text = self.run_mode_combo.currentText()
            run_mode = "iter" if "Iteration" in run_mode_text else "nfe"

            max_iters = int(self.max_iter_input.text())
            max_nfe = int(self.max_nfe_input.text())
            pop_size = int(self.pop_size_input.text())

            if validate:
                if run_mode == "iter" and max_iters <= 0:
                    raise ValueError("Số vòng lặp (Max Iter) phải lớn hơn 0.")
                if run_mode == "nfe" and max_nfe <= 0:
                    raise ValueError(
                        "Số lần gọi hàm (Max NFE) phải lớn hơn 0.")
                if pop_size <= 0:
                    raise ValueError(
                        "Kích thước quần thể (Pop Size) phải lớn hơn 0.")

            return run_mode, max_iters, max_nfe, pop_size

        except Exception as e:
            if validate:
                QMessageBox.critical(
                    self, "Lỗi Tham số Chung", f"Lỗi không xác định: {e}.")
            return None, None, None, None

    def _get_abc_params(self):
        try:
            limit = int(self.abc_limit_input.text())
            if limit <= 0:
                raise ValueError("Limit phải lớn hơn 0.")
            return limit
        except ValueError as e:
            QMessageBox.critical(self, "Lỗi Tham số ABC",
                                 f"Giá trị Limit không hợp lệ: {e}")
            return None

    def _get_ga_params(self):
        try:
            c_rate = float(self.ga_crossover_input.text())
            m_rate = float(self.ga_mutation_input.text())
            if not (0.0 <= c_rate <= 1.0):
                raise ValueError("Tỷ lệ lai ghép phải từ 0.0 đến 1.0.")
            if not (0.0 <= m_rate <= 1.0):
                raise ValueError("Tỷ lệ đột biến phải từ 0.0 đến 1.0.")
            return c_rate, m_rate
        except ValueError as e:
            QMessageBox.critical(self, "Lỗi Tham số GA",
                                 f"Tham số GA không hợp lệ: {e}")
            return None, None

    def _get_hc_params(self):
        try:
            step_size = float(self.hc_step_size_input.text())
            if step_size <= 0:
                raise ValueError("Step Size phải lớn hơn 0.")
            return step_size
        except ValueError as e:
            QMessageBox.critical(self, "Lỗi Tham số HC",
                                 f"Step Size không hợp lệ: {e}")
            return None

    def _get_sa_params(self):
        try:
            temp = float(self.sa_temp_input.text())
            cooling = float(self.sa_cooling_input.text())
            step_size = float(self.sa_step_size_input.text())
            if temp <= 0:
                raise ValueError("Nhiệt độ ban đầu phải lớn hơn 0.")
            if not (0.0 < cooling < 1.0):
                raise ValueError(
                    "Tỷ lệ làm mát phải nằm trong khoảng (0.0, 1.0).")
            if step_size <= 0:
                raise ValueError("Step Size phải lớn hơn 0.")
            return temp, cooling, step_size
        except ValueError as e:
            QMessageBox.critical(self, "Lỗi Tham số SA",
                                 f"Tham số SA không hợp lệ: {e}")
            return None, None, None

    def _get_fa_params(self):
        try:
            alpha = float(self.fa_alpha_input.text())
            beta0 = float(self.fa_beta0_input.text())
            gamma = float(self.fa_gamma_input.text())
            if not (0.0 <= alpha <= 1.0):
                raise ValueError("Alpha phải nằm trong khoảng [0.0, 1.0].")
            if beta0 <= 0:
                raise ValueError("Beta0 phải lớn hơn 0.")
            if gamma <= 0:
                raise ValueError("Gamma phải lớn hơn 0.")
            return alpha, beta0, gamma
        except ValueError as e:
            QMessageBox.critical(self, "Lỗi Tham số FA",
                                 f"Tham số FA không hợp lệ: {e}")
            return None, None, None

    def _get_pso_params(self):
        try:
            w = float(self.pso_w_input.text())
            c1 = float(self.pso_c1_input.text())
            c2 = float(self.pso_c2_input.text())
            if not (0.0 <= w <= 1.0):
                raise ValueError(
                    "Quán tính (w) phải nằm trong khoảng [0.0, 1.0].")
            if c1 < 0 or c2 < 0:
                raise ValueError("Hệ số c1 và c2 phải là số dương.")
            return w, c1, c2
        except ValueError as e:
            QMessageBox.critical(self, "Lỗi Tham số PSO",
                                 f"Tham số PSO không hợp lệ: {e}")
            return None, None, None

    def _get_cs_params(self):
        try:
            pa = float(self.cs_pa_input.text())
            alpha = float(self.cs_alpha_input.text())
            if not (0.0 < pa < 1.0):
                raise ValueError(
                    "pa (xác suất) phải nằm trong khoảng (0.0, 1.0).")
            if alpha <= 0:
                raise ValueError("alpha (hệ số bước) phải lớn hơn 0.")
            return pa, alpha
        except ValueError as e:
            QMessageBox.critical(self, "Lỗi Tham số CS",
                                 f"Tham số CS không hợp lệ: {e}")
            return None, None

    def _set_buttons_enabled(self, enabled):

        self.run_abc_button.setEnabled(enabled)
        self.run_ga_button.setEnabled(enabled)
        self.run_hc_button.setEnabled(enabled)
        self.run_sa_button.setEnabled(enabled)
        self.run_fa_button.setEnabled(enabled)
        self.run_pso_button.setEnabled(enabled)
        self.run_cs_button.setEnabled(enabled)

        if enabled:
            self.run_abc_button.setText("Chạy ABC")
            self.run_ga_button.setText("Chạy GA")
            self.run_hc_button.setText("Chạy HC")
            self.run_sa_button.setText("Chạy SA")
            self.run_fa_button.setText("Chạy FA")
            self.run_pso_button.setText("Chạy PSO")
            self.run_cs_button.setText("Chạy CS")

    def _reset_results_and_plots(self):

        self.history_abc = None
        self.best_abc = None
        self.fitness_abc = None
        self.H_fit_abc = None

        self.history_ga = None
        self.best_ga = None
        self.fitness_ga = None
        self.H_fit_ga = None

        self.history_hc = None
        self.best_hc = None
        self.fitness_hc = None
        self.H_fit_hc = None

        self.history_sa = None
        self.best_sa = None
        self.fitness_sa = None
        self.H_fit_sa = None

        self.history_fa = None
        self.best_fa = None
        self.fitness_fa = None
        self.H_fit_fa = None

        self.history_pso = None
        self.best_pso = None
        self.fitness_pso = None
        self.H_fit_pso = None

        self.history_cs = None
        self.best_cs = None
        self.fitness_cs = None
        self.H_fit_cs = None

        self.abc_fitness_label.setText("ABC Best Fitness: N/A")
        self.abc_solution_label.setText("ABC Best Solution: N/A")
        self.ga_fitness_label.setText("GA Best Fitness: N/A")
        self.ga_solution_label.setText("GA Best Solution: N/A")
        self.hc_fitness_label.setText("HC Best Fitness: N/A")
        self.hc_solution_label.setText("HC Best Solution: N/A")
        self.sa_fitness_label.setText("SA Best Fitness: N/A")
        self.sa_solution_label.setText("SA Best Solution: N/A")
        self.fa_fitness_label.setText("FA Best Fitness: N/A")
        self.fa_solution_label.setText("FA Best Solution: N/A")
        self.pso_fitness_label.setText("PSO Best Fitness: N/A")
        self.pso_solution_label.setText("PSO Best Solution: N/A")
        self.cs_fitness_label.setText("CS Best Fitness: N/A")
        self.cs_solution_label.setText("CS Best Solution: N/A")

        self._stop_all_animations()
        self._plot_convergence(mode='iter')
        self._plot_convergence(mode='nfe')

        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = get_function_bounds(selected_func)
        self._plot_contour(selected_func, l_bound, u_bound)

        canvases = [
            self.canvas_animation_abc, self.canvas_animation_pso,
            self.canvas_animation_fa, self.canvas_animation_cs,
            self.canvas_animation_ga, self.canvas_animation_sa,
            self.canvas_animation_hc
        ]
        names = ["ABC", "PSO", "FA", "CS", "GA", "SA", "HC"]

        for canvas, name in zip(canvases, names):
            ax = canvas.axes
            ax.cla()
            ax.text(0.5, 0.5, f"Vui lòng chạy thuật toán {name}",
                    ha='center', va='center', transform=ax.transAxes)
            canvas.draw()

    def _run_abc(self):
        run_mode, max_iters, max_nfe, pop_size = self._get_parameters()
        limit = self._get_abc_params()
        if run_mode is None or limit is None:
            return

        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = self.current_bounds
        self._set_buttons_enabled(False)
        self.run_abc_button.setText("Đang chạy...")
        self._stop_all_animations()

        try:

            iters_to_run = max_iters if run_mode == "iter" else 5000
            nfe_to_run = 9999999 if run_mode == "iter" else max_nfe

            abc_solver = ArtificialBeeColony(selected_func, l_bound, u_bound,
                                             max_iterations=iters_to_run, max_nfe=nfe_to_run,
                                             num_employed=pop_size // 2, num_onlooker=pop_size // 2, limit=limit)

            if run_mode == "iter":

                self.history_abc, self.best_abc, self.fitness_abc, self.H_fit_abc = abc_solver.run_by_iteration()
            else:

                self.history_abc, self.best_abc, self.fitness_abc, self.H_fit_abc = abc_solver.run_by_nfe()

            self.abc_fitness_label.setText(
                f"ABC Best Fitness: {self.fitness_abc:.6f} (NFE: {abc_solver.nfe})")
            self.abc_solution_label.setText(
                f"ABC Best Solution (X): [{self.best_abc[0]:.4f}, {self.best_abc[1]:.4f}]")

            self._plot_convergence(mode=run_mode)
            self._plot_contour(selected_func, l_bound, u_bound)
            self._plot_animation_2d(
                selected_func, l_bound, u_bound, self.history_abc, "ABC", self.canvas_animation_abc)
            self.tab_widget.setCurrentIndex(4)
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi Tính toán (ABC)", f"Đã xảy ra lỗi: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _run_ga(self):
        run_mode, max_iters, max_nfe, pop_size = self._get_parameters()
        c_rate, m_rate = self._get_ga_params()
        if run_mode is None or c_rate is None:
            return

        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = self.current_bounds
        self._set_buttons_enabled(False)
        self.run_ga_button.setText("Đang chạy...")
        self._stop_all_animations()

        try:
            iters_to_run = max_iters if run_mode == "iter" else 5000
            nfe_to_run = 9999999 if run_mode == "iter" else max_nfe

            ga_solver = GeneticAlgorithm(selected_func, l_bound, u_bound,
                                         max_iterations=iters_to_run, max_nfe=nfe_to_run,
                                         population_size=pop_size, crossover_rate=c_rate, mutation_rate=m_rate)

            if run_mode == "iter":

                self.history_ga, self.best_ga, self.fitness_ga, self.H_fit_ga = ga_solver.run_by_iteration()
            else:

                self.history_ga, self.best_ga, self.fitness_ga, self.H_fit_ga = ga_solver.run_by_nfe()

            self.ga_fitness_label.setText(
                f"GA Best Fitness: {self.fitness_ga:.6f} (NFE: {ga_solver.nfe})")
            self.ga_solution_label.setText(
                f"GA Best Solution (X): [{self.best_ga[0]:.4f}, {self.best_ga[1]:.4f}]")

            self._plot_convergence(mode=run_mode)
            self._plot_contour(selected_func, l_bound, u_bound)
            self._plot_animation_2d(
                selected_func, l_bound, u_bound, self.history_ga, "GA", self.canvas_animation_ga)
            self.tab_widget.setCurrentIndex(8)
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi Tính toán (GA)", f"Đã xảy ra lỗi: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _run_hc(self):
        run_mode, max_iters, max_nfe, _ = self._get_parameters()
        step_size = self._get_hc_params()
        if run_mode is None or step_size is None:
            return

        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = self.current_bounds
        self._set_buttons_enabled(False)
        self.run_hc_button.setText("Đang chạy...")
        self._stop_all_animations()

        try:
            iters_to_run = max_iters if run_mode == "iter" else 5000
            nfe_to_run = 9999999 if run_mode == "iter" else max_nfe

            hc_solver = HillClimbing(selected_func, l_bound, u_bound,
                                     max_iterations=iters_to_run, max_nfe=nfe_to_run,
                                     step_size=step_size)

            if run_mode == "iter":

                self.history_hc, self.best_hc, self.fitness_hc, self.H_fit_hc = hc_solver.run_by_iteration()
            else:

                self.history_hc, self.best_hc, self.fitness_hc, self.H_fit_hc = hc_solver.run_by_nfe()

            self.hc_fitness_label.setText(
                f"HC Best Fitness: {self.fitness_hc:.6f} (NFE: {hc_solver.nfe})")
            self.hc_solution_label.setText(
                f"HC Best Solution (X): [{self.best_hc[0]:.4f}, {self.best_hc[1]:.4f}]")

            self._plot_convergence(mode=run_mode)
            self._plot_contour(selected_func, l_bound, u_bound)
            self._plot_animation_2d(selected_func, l_bound, u_bound, self.history_hc,
                                    "HC", self.canvas_animation_hc, single_point=True)
            self.tab_widget.setCurrentIndex(10)
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi Tính toán (HC)", f"Đã xảy ra lỗi: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _run_sa(self):
        run_mode, max_iters, max_nfe, _ = self._get_parameters()
        temp, cooling, step_size = self._get_sa_params()
        if run_mode is None or temp is None:
            return

        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = self.current_bounds
        self._set_buttons_enabled(False)
        self.run_sa_button.setText("Đang chạy...")
        self._stop_all_animations()

        try:
            iters_to_run = max_iters if run_mode == "iter" else 5000
            nfe_to_run = 9999999 if run_mode == "iter" else max_nfe

            sa_solver = SimulatedAnnealing(selected_func, l_bound, u_bound,
                                           max_iterations=iters_to_run, max_nfe=nfe_to_run,
                                           initial_temp=temp, cooling_rate=cooling, step_size=step_size)

            if run_mode == "iter":

                self.history_sa, self.best_sa, self.fitness_sa, self.H_fit_sa = sa_solver.run_by_iteration()
            else:

                self.history_sa, self.best_sa, self.fitness_sa, self.H_fit_sa = sa_solver.run_by_nfe()

            self.sa_fitness_label.setText(
                f"SA Best Fitness: {self.fitness_sa:.6f} (NFE: {sa_solver.nfe})")
            self.sa_solution_label.setText(
                f"SA Best Solution (X): [{self.best_sa[0]:.4f}, {self.best_sa[1]:.4f}]")

            self._plot_convergence(mode=run_mode)
            self._plot_contour(selected_func, l_bound, u_bound)
            self._plot_animation_2d(selected_func, l_bound, u_bound, self.history_sa,
                                    "SA", self.canvas_animation_sa, single_point=True)
            self.tab_widget.setCurrentIndex(9)
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi Tính toán (SA)", f"Đã xảy ra lỗi: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _run_fa(self):
        run_mode, max_iters, max_nfe, pop_size = self._get_parameters()
        alpha, beta0, gamma = self._get_fa_params()
        if run_mode is None or alpha is None:
            return

        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = self.current_bounds
        self._set_buttons_enabled(False)
        self.run_fa_button.setText("Đang chạy...")
        self._stop_all_animations()

        try:
            iters_to_run = max_iters if run_mode == "iter" else 5000
            nfe_to_run = 9999999 if run_mode == "iter" else max_nfe

            fa_solver = FireflyAlgorithm(selected_func, l_bound, u_bound,
                                         max_iterations=iters_to_run, max_nfe=nfe_to_run,
                                         n_fireflies=pop_size, alpha=alpha, beta0=beta0, gamma=gamma)

            if run_mode == "iter":

                self.history_fa, self.best_fa, self.fitness_fa, self.H_fit_fa = fa_solver.run_by_iteration()
            else:

                self.history_fa, self.best_fa, self.fitness_fa, self.H_fit_fa = fa_solver.run_by_nfe()

            self.fa_fitness_label.setText(
                f"FA Best Fitness: {self.fitness_fa:.6f} (NFE: {fa_solver.nfe})")
            self.fa_solution_label.setText(
                f"FA Best Solution (X): [{self.best_fa[0]:.4f}, {self.best_fa[1]:.4f}]")

            self._plot_convergence(mode=run_mode)
            self._plot_contour(selected_func, l_bound, u_bound)
            self._plot_animation_2d(
                selected_func, l_bound, u_bound, self.history_fa, "FA", self.canvas_animation_fa)
            self.tab_widget.setCurrentIndex(6)
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi Tính toán (FA)", f"Đã xảy ra lỗi: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _run_pso(self):
        run_mode, max_iters, max_nfe, pop_size = self._get_parameters()
        w, c1, c2 = self._get_pso_params()
        if run_mode is None or w is None:
            return

        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = self.current_bounds
        self._set_buttons_enabled(False)
        self.run_pso_button.setText("Đang chạy...")
        self._stop_all_animations()

        try:
            iters_to_run = max_iters if run_mode == "iter" else 5000
            nfe_to_run = 9999999 if run_mode == "iter" else max_nfe

            pso_solver = ParticleSwarmOptimization(selected_func, l_bound, u_bound,
                                                   max_iterations=iters_to_run, max_nfe=nfe_to_run,
                                                   n_particles=pop_size, w=w, c1=c1, c2=c2)

            if run_mode == "iter":

                self.history_pso, self.best_pso, self.fitness_pso, self.H_fit_pso = pso_solver.run_by_iteration()
            else:

                self.history_pso, self.best_pso, self.fitness_pso, self.H_fit_pso = pso_solver.run_by_nfe()

            self.pso_fitness_label.setText(
                f"PSO Best Fitness: {self.fitness_pso:.6f} (NFE: {pso_solver.nfe})")
            self.pso_solution_label.setText(
                f"PSO Best Solution (X): [{self.best_pso[0]:.4f}, {self.best_pso[1]:.4f}]")

            self._plot_convergence(mode=run_mode)
            self._plot_contour(selected_func, l_bound, u_bound)
            self._plot_animation_2d(
                selected_func, l_bound, u_bound, self.history_pso, "PSO", self.canvas_animation_pso)
            self.tab_widget.setCurrentIndex(5)
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi Tính toán (PSO)", f"Đã xảy ra lỗi: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _run_cs(self):
        run_mode, max_iters, max_nfe, pop_size = self._get_parameters()
        pa, alpha = self._get_cs_params()
        if run_mode is None or pa is None:
            return

        selected_func = self.function_map[self.func_combo.currentText()]
        l_bound, u_bound = self.current_bounds
        self._set_buttons_enabled(False)
        self.run_cs_button.setText("Đang chạy...")
        self._stop_all_animations()

        try:
            iters_to_run = max_iters if run_mode == "iter" else 5000
            nfe_to_run = 9999999 if run_mode == "iter" else max_nfe

            cs_solver = CuckooSearch(selected_func, l_bound, u_bound,
                                     max_iterations=iters_to_run, max_nfe=nfe_to_run,
                                     n=pop_size, pa=pa, alpha=alpha)

            if run_mode == "iter":

                self.history_cs, self.best_cs, self.fitness_cs, self.H_fit_cs = cs_solver.run_by_iteration()
            else:

                self.history_cs, self.best_cs, self.fitness_cs, self.H_fit_cs = cs_solver.run_by_nfe()

            self.cs_fitness_label.setText(
                f"CS Best Fitness: {self.fitness_cs:.6f} (NFE: {cs_solver.nfe})")
            self.cs_solution_label.setText(
                f"CS Best Solution (X): [{self.best_cs[0]:.4f}, {self.best_cs[1]:.4f}]")

            self._plot_convergence(mode=run_mode)
            self._plot_contour(selected_func, l_bound, u_bound)
            self._plot_animation_2d(
                selected_func, l_bound, u_bound, self.history_cs, "CS", self.canvas_animation_cs)
            self.tab_widget.setCurrentIndex(7)
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi Tính toán (CS)", f"Đã xảy ra lỗi: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _plot_3d(self):
        selected_func = self.function_map[self.func_combo.currentText()]
        func_name = self.func_combo.currentText()
        l_bound, u_bound = self.current_bounds

        fig = self.canvas_3d.fig
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')

        try:
            x = np.linspace(l_bound, u_bound, 500)
            y = np.linspace(l_bound, u_bound, 500)
            x_meshgrid, y_meshgrid = np.meshgrid(x, y)
            z = selected_func([x_meshgrid, y_meshgrid])

            ax.plot_surface(x_meshgrid, y_meshgrid, z,
                            cmap='viridis', edgecolor='none', alpha=0.8)
            global_min_pos = get_global_min_pos(selected_func)
            if global_min_pos and len(global_min_pos) >= 2:
                global_min_val = selected_func(global_min_pos)
                ax.scatter(global_min_pos[0], global_min_pos[1], global_min_val,
                           color='red', marker='*', s=200, label='Global Minimum', zorder=5)
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_zlabel(f'$f(x)$')
            ax.set_title(f"Đồ thị 3D Hàm {func_name}")
            ax.legend()
            self.canvas_3d.draw()
        except Exception as e:
            ax.set_title("Lỗi vẽ đồ thị 3D")
            print(f"Error plotting 3D: {e}")
            self.canvas_3d.draw()

    def _plot_convergence(self, mode='iter'):

        _, max_iters, max_nfe, _ = self._get_parameters()

        if mode == 'iter':
            canvas = self.canvas_convergence_iter
            xlabel = "Vòng lặp (Iteration)"
            title_suffix = f"(Theo Iteration) - Hàm {self.func_combo.currentText()}"
        else:
            canvas = self.canvas_convergence_nfe
            xlabel = "Số bước lặp (Bị giới hạn bởi NFE)"
            title_suffix = f"(Theo NFE) - Hàm {self.func_combo.currentText()}"

        fig = canvas.fig
        fig.clear()

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        plotted_ax1 = False
        plotted_ax2 = False

        histories_H_fit = [
            (self.H_fit_abc, self.fitness_abc, "ABC"),
            (self.H_fit_pso, self.fitness_pso, "PSO"),
            (self.H_fit_fa, self.fitness_fa, "FA"),
            (self.H_fit_cs, self.fitness_cs, "CS"),
            (self.H_fit_ga, self.fitness_ga, "GA"),
            (self.H_fit_sa, self.fitness_sa, "SA"),
            (self.H_fit_hc, self.fitness_hc, "HC")
        ]

        histories_positions = [
            (self.history_abc, self.fitness_abc, "ABC", False),
            (self.history_pso, self.fitness_pso, "PSO", False),
            (self.history_fa, self.fitness_fa, "FA", False),
            (self.history_cs, self.fitness_cs, "CS", False),
            (self.history_ga, self.fitness_ga, "GA", False),
            (self.history_sa, self.fitness_sa, "SA", True),
            (self.history_hc, self.fitness_hc, "HC", True)
        ]

        selected_func = self.function_map[self.func_combo.currentText()]

        def get_best_of_gen_fitness(history, func, single_point):

            history_to_plot = history[:-1]
            if single_point:
                return [func(positions[0]) for positions in history_to_plot]
            else:
                return [np.min([func(p) for p in positions]) for positions in history_to_plot]

        for fit_hist, best_fit, name in histories_H_fit:
            if fit_hist is not None and len(fit_hist) > 0:
                try:
                    if mode == 'iter':
                        x_axis = np.arange(len(fit_hist))
                    else:
                        x_axis = np.linspace(0, max_nfe, len(fit_hist))
                    ax1.plot(x_axis, fit_hist,
                             label=f'{name} (Best: {best_fit:.4f})',
                             color=self.algo_colors[name])
                    plotted_ax1 = True
                except Exception as e:
                    print(f"Error plotting {name} convergence (ax1): {e}")

        for history, best_fit, name, single_point in histories_positions:
            if history is not None and len(history) > 1:
                try:
                    fit_hist_gen = get_best_of_gen_fitness(
                        history, selected_func, single_point)
                    if mode == 'iter':
                        x_axis = np.arange(len(fit_hist_gen))
                    else:
                        x_axis = np.linspace(0, max_nfe, len(fit_hist_gen))
                    ax2.plot(x_axis, fit_hist_gen,
                             label=f'{name}',
                             color=self.algo_colors[name], linestyle='--')
                    plotted_ax2 = True
                except Exception as e:
                    print(f"Error plotting {name} convergence (ax2): {e}")

        if not plotted_ax1:
            ax1.text(0.5, 0.5, "Chưa có dữ liệu (Best-So-Far)",
                     ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("1. Lịch sử Tốt nhất (Best-So-Far)")
        ax1.set_ylabel("Best Fitness")
        if plotted_ax1:
            ax1.legend()
        ax1.grid(True)
        plt.setp(ax1.get_xticklabels(), visible=False)

        if not plotted_ax2:
            ax2.text(0.5, 0.5, "Chưa có dữ liệu (Best-of-Generation)",
                     ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("2. Fitness Tốt nhất của Thế hệ Hiện tại")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Min Fitness (Gen)")
        if plotted_ax2:
            ax2.legend()
        ax2.grid(True)

        fig.suptitle(f"Đồ thị Hội tụ {title_suffix}", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        canvas.draw()

    def _plot_contour(self, selected_func, l_bound, u_bound):

        fig = self.canvas_contour.fig
        fig.clear()
        ax = fig.add_subplot(111)

        x = np.linspace(l_bound, u_bound, 100)
        y = np.linspace(l_bound, u_bound, 100)
        x_meshgrid, y_meshgrid = np.meshgrid(x, y)
        z = selected_func([x_meshgrid, y_meshgrid])
        ax.contourf(x_meshgrid, y_meshgrid, z,
                    levels=50, cmap='viridis', alpha=0.8)

        best_results = [
            (self.best_abc, 'o', 'yellow', 'ABC'),
            (self.best_pso, 'p', 'lime', 'PSO'),
            (self.best_fa, 'D', 'orange', 'FA'),
            (self.best_cs, 'x', 'magenta', 'CS'),
            (self.best_ga, 's', 'red', 'GA'),
            (self.best_sa, 'v', 'green', 'SA'),
            (self.best_hc, '^', 'cyan', 'HC')
        ]

        for best_pos, marker, color, name in best_results:
            if best_pos is not None:

                if marker == 'x':
                    ax.plot(best_pos[0], best_pos[1], marker, color=color, markersize=10,
                            markeredgecolor='black', label=f'Best ({name})', mew=2)
                else:
                    ax.plot(best_pos[0], best_pos[1], marker, color=color,
                            markersize=10, markeredgecolor='black', label=f'Best ({name})')

        global_min_pos = get_global_min_pos(selected_func)
        if global_min_pos and len(global_min_pos) >= 2:
            ax.plot(global_min_pos[0], global_min_pos[1], 'w*',
                    markersize=12, markeredgecolor='black', label='Global Minimum')

        ax.set_xlim(l_bound, u_bound)
        ax.set_ylim(l_bound, u_bound)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(f"Kết quả cuối trên Biểu đồ Đường Đồng Mức")
        ax.legend(loc='upper right')
        self.canvas_contour.draw()

    def _plot_animation_2d(self, func, l_bound, u_bound, positions_history, algorithm_name, canvas, single_point=False):

        fig = canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)

        if positions_history is None or len(positions_history) == 0:
            ax.text(0.5, 0.5, f"Lịch sử vị trí của {algorithm_name} trống.",
                    ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return

        if self.current_animation and self.current_animation._fig == canvas.fig:
            self.current_animation.event_source.stop()

        x = np.linspace(l_bound, u_bound, 100)
        y = np.linspace(l_bound, u_bound, 100)
        x_meshgrid, y_meshgrid = np.meshgrid(x, y)
        z = func([x_meshgrid, y_meshgrid])
        ax.contourf(x_meshgrid, y_meshgrid, z,
                    levels=50, cmap='viridis', alpha=0.8)

        global_min_pos = get_global_min_pos(func)
        if global_min_pos and len(global_min_pos) >= 2:
            ax.plot(global_min_pos[0], global_min_pos[1], 'w*',
                    markersize=12, markeredgecolor='black', label='Global Minimum')

        if single_point:

            scat, = ax.plot([], [], 'o', color='yellow', markersize=8,
                            label=f'Giải pháp ({algorithm_name})', markeredgecolor='black')
            best_point = None
        else:

            scat = ax.scatter([], [], c='yellow', s=50,
                              label=f'Quần thể ({algorithm_name})', edgecolor='black')
            best_point, = ax.plot([], [], 'o', color='red', markersize=8,
                                  markeredgecolor='black', label='Best Solution Found')

        max_iter = len(positions_history) - 1
        iteration_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                 color='black', verticalalignment='top', fontsize=10)
        ax.set_xlim(l_bound, u_bound)
        ax.set_ylim(l_bound, u_bound)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(
            f"Animation {algorithm_name} - Hàm {self.func_combo.currentText()}")
        ax.legend(loc='upper right')

        def update(frame):
            current_positions = positions_history[frame]

            if single_point:

                scat.set_data(np.array([current_positions[0][0]]), np.array(
                    [current_positions[0][1]]))
                return_artists = [scat, iteration_text]
            else:

                scat.set_offsets(current_positions)

                current_fitness = np.array([func(p)
                                           for p in current_positions])
                current_best_pos = current_positions[np.argmin(
                    current_fitness)]
                best_point.set_data(
                    np.array([current_best_pos[0]]), np.array([current_best_pos[1]]))
                return_artists = [scat, best_point, iteration_text]

            iteration_text.set_text(f"Iteration: {frame}/{max_iter}")
            return return_artists

        self.current_animation = FuncAnimation(
            canvas.fig, update, frames=len(positions_history),
            interval=150, repeat=True, blit=False
        )
        canvas.draw()


if __name__ == '__main__':

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    window = AppWindow()
    window.show()
    sys.exit(app.exec_())
