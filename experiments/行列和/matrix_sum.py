from typing import Callable
import numpy as np
from numba import njit
import time


@njit(cache=True)
def repeat_matrix_sum_jitted(matrix: np.ndarray, n_repeat: int) -> np.ndarray:
    # 計測と高速化対象
    m = matrix
    for n in range(n_repeat):
        m += matrix
    return m


def repeat_matrix_sum(matrix: np.ndarray, n_repeat: int) -> np.ndarray:
    # 計測と高速化対象
    m = matrix
    for n in range(n_repeat):
        m += matrix
    return m


def experiments(experiment_sizes: list[tuple[int, int]], repeat_matrix_sum: Callable):
    perf_results = {}
    for n_size, n_iter in experiment_sizes:
        matrix = np.random.normal(loc=0.0, scale=0.01, size=(n_size, n_size))
        start = time.perf_counter()
        ret = repeat_matrix_sum(matrix=matrix, n_repeat=n_iter)
        end = time.perf_counter()  # 計測終了

        perf_results[(n_size, n_iter)] = end - start
        print(f"{perf_results[(n_size, n_iter)]=:.2f} sec.")
