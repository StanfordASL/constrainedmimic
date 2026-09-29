"""Utilities for timing tests"""

import time
from typing import Callable, Tuple

import jax


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")


def benchmark_function(
    func: Callable, args: Tuple, n_calls: int = 100000, n_warmup: int = 10
) -> Tuple[float, float]:
    """Benchmark the timing performance of a JAX function

    Args:
        func (Callable): JAX function to test
        args (Tuple): Example arguments for the function
        n_calls (int, optional): Number of calls to use for timing. Defaults to 100000.
        n_warmup (int, optional): Number of 'warmup' calls before timing. Defaults to 10.

    Returns:
        Tuple[float, float]:
            avg_time (float): Average time per function call after JIT
            jit_time (float): Time taken to JIT the function
    """
    func_jit = jax.jit(func)

    # JIT timing
    start_time = time.perf_counter()
    res = func_jit(*args)
    jax.block_until_ready(res)
    jit_time = time.perf_counter() - start_time

    # Warmup
    for _ in range(n_warmup):
        res = func_jit(*args)
        jax.block_until_ready(res)

    # Timing
    start = time.perf_counter()
    for _ in range(n_calls):
        res = func_jit(*args)
        jax.block_until_ready(res)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / n_calls

    return avg_time, jit_time
