from typing import Callable
import numpy as np
from numba import njit
import time
import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def matmul_jax2(matrix1: jnp.ndarray, matrix2: jnp.ndarray) -> jnp.ndarray:
    return matrix1.dot(matrix2)


def repeat_matrix_products_jaxjitted2(matrix: np.ndarray, n_repeat: int) -> np.ndarray:
    matrix_jax = jnp.array(matrix)
    m = matrix_jax
    for n in range(n_repeat):
        m = matmul_jax2(m, matrix_jax)
        m.block_until_ready()
    return m


@partial(jax.jit, static_argnums=1)
def matmul_jax(matrix: jnp.ndarray, n_repeat: int) -> jnp.ndarray:
    # これではメモリ不足になる
    m = matrix
    for n in range(n_repeat):
        m = m.dot(matrix)
    return m


def repeat_matrix_products_jaxjitted(matrix: np.ndarray, n_repeat: int) -> np.ndarray:
    matrix_jax = jnp.array(matrix)
    ret = matmul_jax(matrix_jax, n_repeat)
    ret.block_until_ready()
    return ret


def repeat_matrix_products_jax(matrix: np.ndarray, n_repeat: int) -> np.ndarray:
    matrix_jax = jnp.array(matrix)
    m = matrix_jax
    for n in range(n_repeat):
        m = m.dot(matrix_jax)
    m.block_until_ready()
    return m


@njit(cache=True)
def repeat_matrix_products_jitted(matrix: np.ndarray, n_repeat: int) -> np.ndarray:
    # 計測と高速化対象
    m = matrix
    for n in range(n_repeat):
        # m = np.dot(m, matrix)  # scipy 0.16+ is required for linear algebra
        m = m.dot(matrix)
    return m


def repeat_matrix_products(matrix: np.ndarray, n_repeat: int) -> np.ndarray:
    # 計測と高速化対象
    m = matrix
    for n in range(n_repeat):
        m = np.dot(m, matrix)
    return m


def experiments(experiment_sizes: list[tuple[int, int]], repeat_matrix_products: Callable):
    perf_results = {}
    for n_size, n_iter in experiment_sizes:
        matrix = np.random.normal(loc=0.0, scale=0.01, size=(n_size, n_size))
        start = time.perf_counter()
        ret = repeat_matrix_products(matrix=matrix, n_repeat=n_iter)
        end = time.perf_counter()  # 計測終了

        perf_results[(n_size, n_iter)] = end - start
        print(f"{perf_results[(n_size, n_iter)]=:.2f} sec.")
