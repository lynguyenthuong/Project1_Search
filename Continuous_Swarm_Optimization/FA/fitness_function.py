import numpy as np

def rosenbrock_function(x):
    """Hàm Rosenbrock

    Tham số:
        x (List hoặc np.array): Điểm (có thể là bất kỳ chiều nào)

    Trả về:
        float hoặc np.array: Giá trị của hàm Rosenbrock
    """
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2, axis=0)


def sphere_function(x):
    """Hàm Sphere

    Tham số:
        x (List hoặc np.array): Điểm

    Trả về:
        float hoặc np.array: Giá trị của hàm Sphere
    """
    x = np.array(x)
    return np.sum(x**2, axis=0)


def rastrigin_function(x):
    """Hàm Rastrigin

    Tham số:
        x (List hoặc np.array): Điểm

    Trả về:
        float hoặc np.array: Giá trị của hàm Rastrigin
    """
    x = np.array(x)
    d = len(x)
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=0)


def ackley_function(x):
    """Hàm Ackley

    Tham số:
        x (List hoặc np.array): Điểm

    Trả về:
        float hoặc np.array: Giá trị của hàm Ackley
    """
    x = np.array(x)
    d = len(x)
    
    # Các hằng số
    a = 20
    b = 0.2
    c = 2 * np.pi

    sum1 = np.sum(x**2, axis=0)
    sum2 = np.sum(np.cos(c * x), axis=0)

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    return term1 + term2 + a + np.exp(1)


def get_function_bounds(func):
    """Trả về bound thích hợp cho các hàm fitness."""
    func_name = func.__name__

    if func_name == 'ackley_function':
        return -32.768, 32.768
    elif func_name == 'rosenbrock_function':
        return -2.048, 2.048
    elif func_name == 'rastrigin_function':
        return -5.12, 5.12
    elif func_name == 'sphere_function':
        return -5.12, 5.12
    else:
        # Default bounds
        return -10, 10


def get_global_min_pos(func, d=2):
    """Vị trí của global minimum"""
    func_name = func.__name__
    
    if func_name == 'rosenbrock_function':
        return tuple([1] * d)
    elif func_name in ['ackley_function', 'rastrigin_function', 'sphere_function']:
        return tuple([0] * d)
    else:
        return None
