"""Test cases for loading ONNX checkpoints to JAX"""

import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort

from cm_control.twist2_utils.jax_twist2 import load_policy, ONNX_CHECKPOINT

jax.config.update("jax_enable_x64", True)


class TestTwist2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jax_model, cls.jax_params = load_policy()

    def test_onnx_vs_jax_output(self):
        num_tests = 10
        np.random.seed(0)

        # Set up ONNX CPU inference session
        ort_sess = ort.InferenceSession(
            ONNX_CHECKPOINT, providers=["CPUExecutionProvider"]
        )
        input_name = ort_sess.get_inputs()[0].name

        for _ in range(num_tests):
            dummy_input_np = np.random.randn(1, 1432).astype(np.float32)
            dummy_input_jax = jnp.asarray(dummy_input_np)
            # Run ONNX Inference
            onnx_out = ort_sess.run(None, {input_name: dummy_input_np})[0]
            # Run JAX Inference
            jax_out = self.jax_model.apply(
                self.jax_params, dummy_input_jax, train=False
            )
            # Compare results (with slight tolerance)
            is_close = np.allclose(onnx_out, jax_out, atol=1e-5)
            self.assertTrue(is_close)

    def test_jax_speed(self):
        @jax.jit
        def call_model(x):
            return self.jax_model.apply(self.jax_params, x, train=False)

        num_tests = 100
        np.random.seed(0)
        times = []
        for _ in range(num_tests):
            x = np.random.randn(1, 1432)  # .astype(np.float32)
            start_time = time.perf_counter()
            _ = call_model(x).block_until_ready()
            times.append(time.perf_counter() - start_time)

        jit_time = times[0]
        avg_time = np.mean(times[1:])
        avg_time_ms = avg_time * 1e3
        print("\nTwist2:")
        print("JIT time (seconds): ", jit_time)
        print("Average inference time (milliseconds): ", avg_time_ms)


if __name__ == "__main__":
    unittest.main()
