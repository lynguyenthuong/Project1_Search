# Project1_Search
Nhóm chúng em xin hướng dẫn cách thầy kiểm tra lại các thuật toán của chúng em. 

1. Yêu cầu về các thư viện: 
    - numpy
    - pandas
    - matplotlib
    - sys
    - PyQt5
    - math
    - mpl_toolkits
    - utils
    - os
    - dataclasses
    - typing
    - IPython (nếu thầy dùng Jupyter Notebook để chạy)

2. Source code của chúng em có định dạng như sau:
    Project1_Search
        Continuous_Swarm_Optimization
            ABC
                ABC.py
            App
                ...
                app.py
            CS
                CS.py
            FA
                ...
                FA.py
            PSO
                ...
                PSO.py
        Discrete_Swarm_Optimization
            ...
            tsp_app.py

3. Để kiểm tra cụ thể các yêu cầu trong đồ án cho các thuật toán tối ưu hóa LIÊN TỤC, mời thầy truy cập các file sau:
    - Đối với thuật toán Artificial Bee Colony:         Project1_Search\Continuous_Swarm_Optimization\ABC\ABC.py
    - Đối với thuật toán Cuckoo Search:                 Project1_Search\Continuous_Swarm_Optimization\CS\CS.py
    - Đối với thuật toán Firefly Algorithm:             Project1_Search\Continuous_Swarm_Optimization\FA\FA.py
    - Đối với thuật toán Particle Swarm Optimization:   Project1_Search\Continuous_Swarm_Optimization\PSO\PSO.py

4. Để kiểm tra cụ thể các yêu cầu trong đồ án cho các thuật toán tối ưu hóa RỜI RẠC, mời thầy truy cập các file sau:
    - Đối với thuật toán Ant Colony Optimization:       Project1_Search\Discrete_Swarm_Optimization\tsp_app.py

5. Để kiểm tra phần demo trong video, mời thầy truy cập các file sau:
    - Đối với các thuật toán tối ưu hóa liên tục:       Project1_Search\Continuous_Swarm_Optimization\App\app.py
    - Đối với các thuật toán tối ưu hóa rời rạc:        Project1_Search\Discrete_Swarm_Optimization\tsp_app.py

6. Lưu ý trong quá trình test thuật toán Ant Colony Optimization: Nhập đường dẫn ma trận (ví dụ `Discrete_Swarm_Optimization/data/weights_25.csv`) rồi bấm “Nạp ma trận”. Nếu muốn thay đổi sang ma trận 10 hay 50 đỉnh thì chỉ cần đổi số 25 trong đường dẫn thành 10 hoặc 50, rồi bấm "Nạp ma trận".

7. Phần data và test case thì chúng em đã chuẩn bị sẵn trong các folder chung với các thuật toán

8. Link youtube: https://www.youtube.com/watch?v=DxfBZdInzoU
