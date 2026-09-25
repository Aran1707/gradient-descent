"""Unit and numerical verification tests for seminar optimizers and landscapes.

Implements Stage 0 checks and Phase B verification from PROJECT_PLAN.md:
1. Analytical gradients vs finite-difference gradients for all 4 objectives.
2. Sphere exact single-step test (matching PROJECT_PLAN.md section 7.2 & MATH_NOTES.md).
3. Ill-conditioned stability boundary (0 < eta < 0.04).
4. Correctness of all 8 2D optimizers.
5. Parameter shape preservation and state isolation in src/optimizers/.
6. Adam refactor numerical verification.
7. Sharpness-Aware Minimization (SAM) perturbation and restoration mechanics.
8. Tiny CNN forward/backward/step integration smoke test.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from landscapes.objectives import OBJECTIVES, ill_conditioned, sphere
from landscapes.optimizers import (
    SGD as ToySGD,
)
from landscapes.optimizers import (
    AdaGrad as ToyAdaGrad,
)
from landscapes.optimizers import (
    Adam as ToyAdam,
)
from landscapes.optimizers import (
    AdamW as ToyAdamW,
)
from landscapes.optimizers import (
    Lion as ToyLion,
)
from landscapes.optimizers import (
    Momentum as ToyMomentum,
)
from landscapes.optimizers import (
    Nesterov as ToyNesterov,
)
from landscapes.optimizers import (
    RMSProp as ToyRMSProp,
)
from landscapes.optimizers import (
    build_toy_optimizer,
)
from landscapes.optimizers import (
    run as run_toy,
)
from optimizers import (
    SAM,
    SGD,
    Adam,
    AdamW,
    Lion,
    build_optimizer,
    build_sam_optimizer,
    optimizer_names,
)
from user_numpy_cnn.train import CNNModel


def _numerical_gradient(fn, theta, eps=1e-6):
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        grad[i] = (fn(theta_plus) - fn(theta_minus)) / (2 * eps)
    return grad


class TestLandscapeGradients(unittest.TestCase):
    """Verify analytical gradients against finite differences for all 4 landscapes."""

    def test_sphere_gradient(self):
        obj, grad_fn = OBJECTIVES["sphere"]
        test_points = [[0.0, 0.0], [1.0, 2.0], [-3.0, 4.5], [0.5, -1.2]]
        for pt in test_points:
            analytical = grad_fn(pt)
            numerical = _numerical_gradient(obj, pt)
            np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-5)

    def test_ill_conditioned_gradient(self):
        obj, grad_fn = OBJECTIVES["ill-conditioned"]
        test_points = [[0.0, 0.0], [2.8, 1.0], [-1.5, 0.8], [3.2, -0.4]]
        for pt in test_points:
            analytical = grad_fn(pt)
            numerical = _numerical_gradient(obj, pt)
            np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-5)

    def test_double_well_gradient(self):
        obj, grad_fn = OBJECTIVES["double-well"]
        test_points = [[-1.0, 0.0], [1.0, 0.0], [0.0, 0.5], [-2.0, 1.5]]
        for pt in test_points:
            analytical = grad_fn(pt)
            numerical = _numerical_gradient(obj, pt)
            np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-5)

    def test_saddle_gradient(self):
        obj, grad_fn = OBJECTIVES["saddle"]
        test_points = [[0.0, 0.0], [1.0, 1.0], [-2.0, 3.0], [0.5, -0.5]]
        for pt in test_points:
            analytical = grad_fn(pt)
            numerical = _numerical_gradient(obj, pt)
            np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-5)


class TestMathematicalPredictions(unittest.TestCase):
    """Check mathematical seminar derivations against actual optimizer steps."""

    def test_sphere_step_exact(self):
        """PROJECT_PLAN 7.2: theta_0 = (1, 2), eta = 0.1 -> theta_1 = (0.8, 1.6), loss 5 -> 3.2."""
        _, grad_fn = OBJECTIVES["sphere"]
        theta_0 = np.array([1.0, 2.0])
        opt = ToySGD(lr=0.1)
        theta_1 = opt.step(theta_0, grad_fn(theta_0))

        expected = np.array([0.8, 1.6])
        np.testing.assert_allclose(theta_1, expected, rtol=1e-6)
        self.assertAlmostEqual(sphere(theta_1), 3.2, places=6)

    def test_ill_conditioned_stability_boundary(self):
        """PROJECT_PLAN 7.4: 0 < eta < 0.04 converges; eta >= 0.04 oscillates/diverges in y."""
        _, grad_fn = OBJECTIVES["ill-conditioned"]
        start = np.array([2.8, 1.0])

        # Stable step: eta = 0.03 < 0.04
        opt_stable = ToySGD(lr=0.03)
        traj_stable = run_toy(opt_stable, start, grad_fn, steps=30)
        self.assertLess(abs(traj_stable[-1][1]), abs(start[1]))
        self.assertLess(ill_conditioned(traj_stable[-1]), ill_conditioned(start))

        # Divergent step: eta = 0.05 > 0.04: |1 - 50 * 0.05| = 1.5 > 1, oscillates and grows
        opt_unstable = ToySGD(lr=0.05)
        traj_unstable = run_toy(opt_unstable, start, grad_fn, steps=10)
        self.assertGreater(abs(traj_unstable[-1][1]), abs(start[1]))


class TestToyOptimizers(unittest.TestCase):
    """Verify all 8 2D optimizers in src/landscapes/optimizers.py."""

    def test_all_toy_optimizers_descend_on_sphere(self):
        start = np.array([2.0, 2.0])
        initial_loss = sphere(start)
        _, grad_fn = OBJECTIVES["sphere"]

        opts = [
            ToySGD(0.1),
            ToyMomentum(0.05),
            ToyNesterov(0.05),
            ToyAdaGrad(0.5),
            ToyRMSProp(0.1),
            ToyAdam(0.1),
            ToyAdamW(0.1, weight_decay=0.01),
            ToyLion(0.05),
        ]

        for opt in opts:
            traj = run_toy(opt, start, grad_fn, steps=25)
            final_loss = sphere(traj[-1])
            self.assertLess(
                final_loss,
                initial_loss,
                f"{type(opt).__name__} failed to decrease loss on sphere",
            )

    def test_build_toy_optimizer(self):
        for name in ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adam", "adamw", "lion"]:
            opt = build_toy_optimizer(name, lr=0.01)
            self.assertIsNotNone(opt)


class TestCNNOptimizers(unittest.TestCase):
    """Verify implementations in src/optimizers/."""

    def test_shape_preservation(self):
        """All optimizers must preserve parameter shapes and types."""
        for name in optimizer_names():
            if name.startswith("sam"):
                continue
            opt = build_optimizer(name, lr=1e-3)
            p = np.random.randn(4, 5).astype(np.float32)
            g = np.random.randn(4, 5).astype(np.float32)
            p_orig = p.copy()

            opt.step(((p, g, "param_0"),))
            self.assertEqual(p.shape, p_orig.shape)
            self.assertEqual(p.dtype, np.float32)
            self.assertFalse(np.array_equal(p, p_orig))

    def test_adam_step_formula(self):
        """Validate Adam step against analytical formulas."""
        lr = 1e-2
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        opt = Adam(lr=lr, beta1=beta1, beta2=beta2, eps=eps)

        p = np.array([1.0, 2.0], dtype=np.float32)
        g = np.array([0.5, -0.2], dtype=np.float32)

        opt.step(((p, g, "test_p"),))

        # Expected step 1
        m_1 = (1 - beta1) * g
        v_1 = (1 - beta2) * (g * g)
        m_hat = m_1 / (1 - beta1)
        v_hat = v_1 / (1 - beta2)
        expected_update = lr * m_hat / (np.sqrt(v_hat) + eps)
        expected_p = np.array([1.0, 2.0], dtype=np.float32) - expected_update

        np.testing.assert_allclose(p, expected_p, rtol=1e-5, atol=1e-5)

    def test_adamw_weight_decay(self):
        """AdamW decouples weight decay: p <- p * (1 - lr * wd) - Adam_update."""
        lr = 1e-3
        wd = 0.1
        opt = AdamW(lr=lr, weight_decay=wd)
        p = np.array([1.0, 1.0], dtype=np.float32)
        g = np.zeros_like(p)

        opt.step(((p, g, "w"),))
        expected_p = np.array([1.0, 1.0], dtype=np.float32) * (1.0 - lr * wd)
        np.testing.assert_allclose(p, expected_p, rtol=1e-5)

    def test_lion_sign_step(self):
        """Lion update direction is given by sign of momentum interpolation."""
        lr = 0.05
        opt = Lion(lr=lr, beta1=0.9, beta2=0.99)
        p = np.array([1.0, 1.0], dtype=np.float32)
        g = np.array([2.5, -0.1], dtype=np.float32)

        opt.step(((p, g, "w"),))
        # Initial momentum is 0 -> c = (1 - 0.9) * g = 0.1 * g -> sign(c) = [1.0, -1.0]
        # p <- p - lr * sign(c) = [1.0 - 0.05, 1.0 - (-0.05)] = [0.95, 1.05]
        expected_p = np.array([0.95, 1.05], dtype=np.float32)
        np.testing.assert_allclose(p, expected_p, rtol=1e-5)

    def test_sam_perturb_restore(self):
        """SAM must perturb parameters by rho * g / ||g|| in first_step, and restore in second_step."""
        base_opt = SGD(lr=0.1)
        rho = 0.05
        sam = SAM(base_opt, rho=rho)

        p = np.array([3.0, 4.0], dtype=np.float32)
        g1 = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5.0
        p_orig = p.copy()

        # Step 1: perturbation
        sam.first_step(((p, g1, "param"),))
        norm_g = np.linalg.norm(g1)
        expected_pert = rho * g1 / norm_g  # 0.05 * [3/5, 4/5] = [0.03, 0.04]
        np.testing.assert_allclose(p, p_orig + expected_pert, rtol=1e-5)

        # Step 2: restore and update with new gradient g2
        g2 = np.array([1.0, 1.0], dtype=np.float32)
        sam.second_step(((p, g2, "param"),))
        # After restoration p becomes p_orig, then SGD update: p_orig - 0.1 * g2
        expected_final = p_orig - 0.1 * g2
        np.testing.assert_allclose(p, expected_final, rtol=1e-5)

    def test_sam_factory_helpers(self):
        """SAM factory helper should handle SAM aliases and existing SAM instances cleanly."""
        # Alias "sam" maps to base SGD
        sam_default = build_sam_optimizer("sam")
        self.assertIsInstance(sam_default, SAM)
        self.assertIsInstance(sam_default.base_optimizer, SGD)

        # Alias "sam-sgd" maps to base SGD
        sam_sgd = build_sam_optimizer("sam-sgd")
        self.assertIsInstance(sam_sgd, SAM)
        self.assertIsInstance(sam_sgd.base_optimizer, SGD)

        # Alias "sam-adam" maps to base Adam
        sam_adam = build_sam_optimizer("sam-adam")
        self.assertIsInstance(sam_adam, SAM)
        self.assertIsInstance(sam_adam.base_optimizer, Adam)

        # Direct optimizer name "adam" maps to base Adam
        sam_adam2 = build_sam_optimizer("adam")
        self.assertIsInstance(sam_adam2, SAM)
        self.assertIsInstance(sam_adam2.base_optimizer, Adam)

        # Passing existing SAM un-wraps cleanly without nesting SAM inside SAM
        existing_sam = build_optimizer("sam", lr=1e-3)
        sam_wrapped = build_sam_optimizer(existing_sam)
        self.assertIsInstance(sam_wrapped, SAM)
        self.assertIsInstance(sam_wrapped.base_optimizer, SGD)
        self.assertNotIsInstance(sam_wrapped.base_optimizer, SAM)


class TestCNNSmoke(unittest.TestCase):
    """Test tiny forward, backward, and optimization step on NumPy CNN."""

    def test_forward_backward_step_smoke(self):
        np.random.seed(0)
        model = CNNModel(use_dropout=False)
        X = np.random.randn(4, 1, 28, 28).astype(np.float32)
        y = np.array([0, 1, 2, 3], dtype=np.int64)

        # Forward
        logits = model.forward(X)
        self.assertEqual(logits.shape, (4, 10))
        loss_before = model.loss_fn.forward(logits, y)
        self.assertTrue(np.isfinite(loss_before))

        # Backward
        model.backward(y)

        # Verify gradients exist and are finite
        params = list(model.parameters())
        self.assertGreater(len(params), 0)
        for param, grad, name in params:
            self.assertEqual(param.shape, grad.shape)
            self.assertTrue(np.all(np.isfinite(grad)), f"Non-finite gradient in {name}")

        # Optimizer step
        optimizer = build_optimizer("adam", lr=1e-3)
        model.step(optimizer)

        # Verify next forward pass runs and loss is finite
        logits_after = model.forward(X)
        loss_after = model.loss_fn.forward(logits_after, y)
        self.assertTrue(np.isfinite(loss_after))

    def test_sam_cnn_step_smoke(self):
        """Test two-pass SAM update on the CNN model."""
        np.random.seed(42)
        model = CNNModel(use_dropout=False)
        X = np.random.randn(4, 1, 28, 28).astype(np.float32)
        y = np.array([1, 2, 0, 3], dtype=np.int64)

        base_opt = build_optimizer("sgd", lr=1e-2)
        sam_opt = SAM(base_opt, rho=0.05)

        # Pass 1
        logits1 = model.forward(X)
        _ = model.loss_fn.forward(logits1, y)
        model.backward(y)
        sam_opt.first_step(model.parameters())

        # Pass 2
        logits2 = model.forward(X)
        _ = model.loss_fn.forward(logits2, y)
        model.backward(y)
        sam_opt.second_step(model.parameters())

        # After SAM update, model can forward pass cleanly
        logits3 = model.forward(X)
        loss3 = model.loss_fn.forward(logits3, y)
        self.assertTrue(np.isfinite(loss3))

    def test_evaluate_loss_and_acc(self):
        from user_numpy_cnn.train import evaluate, evaluate_loss_and_acc
        model = CNNModel(use_dropout=False)
        X = np.random.randn(8, 1, 28, 28).astype(np.float32)
        y = np.random.randint(0, 10, size=8)

        loss, acc = evaluate_loss_and_acc(model, X, y, batch_size=4)
        acc_legacy = evaluate(model, X, y, batch_size=4)

        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertEqual(acc, acc_legacy)

    def test_multi_seed_experiments_artifacts(self):
        """Multi-seed experiments should save distinct per-seed files and aggregate metrics."""
        import json
        import tempfile
        from unittest.mock import patch

        from user_numpy_cnn.train import run_multi_seed_experiments

        dummy_x_train = np.random.randn(8, 1, 28, 28).astype(np.float32)
        dummy_y_train = np.random.randint(0, 10, size=8)
        dummy_x_test = np.random.randn(4, 1, 28, 28).astype(np.float32)
        dummy_y_test = np.random.randint(0, 10, size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            model_path = tmppath / "model.npz"
            metrics_path = tmppath / "metrics.json"

            with patch(
                "user_numpy_cnn.train.load_mnist",
                return_value=(dummy_x_train, dummy_y_train, dummy_x_test, dummy_y_test),
            ):
                summary = run_multi_seed_experiments(
                    seeds=(0, 1),
                    model_path=model_path,
                    metrics_path=metrics_path,
                    epochs=1,
                    batch_size=4,
                    val_size=0,
                )

            # Check per-seed model artifacts exist
            self.assertTrue((tmppath / "model_seed_0.npz").exists())
            self.assertTrue((tmppath / "model_seed_1.npz").exists())

            # Check per-seed metrics artifacts exist
            self.assertTrue((tmppath / "metrics_seed_0.json").exists())
            self.assertTrue((tmppath / "metrics_seed_1.json").exists())

            # Check aggregate metrics artifact exists and has correct structure
            self.assertTrue(metrics_path.exists())
            with metrics_path.open("r", encoding="utf-8") as f:
                saved_summary = json.load(f)

            self.assertEqual(saved_summary["seeds"], [0, 1])
            self.assertIn("mean_test_acc", saved_summary)
            self.assertIn("std_test_acc", saved_summary)
            self.assertIn("mean_test_loss", saved_summary)
            self.assertIn("std_test_loss", saved_summary)
            self.assertEqual(len(saved_summary["results"]), 2)
            self.assertEqual(summary["mean_test_acc"], saved_summary["mean_test_acc"])


class TestTrajectoryExport(unittest.TestCase):
    """Test trajectory export script logic."""

    def test_export_trajectories_generates_valid_csv(self):
        import csv
        import tempfile

        from scripts.export_trajectories import TRAJECTORY_CONFIGS, export_trajectory

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            cfg = TRAJECTORY_CONFIGS["gd"]
            meta = export_trajectory("gd", cfg, "ill-conditioned", (2.8, 1.0), tmppath)
            csv_path = tmppath / meta["file"]

            self.assertTrue(csv_path.exists())
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header, ["step", "x", "y", "loss"])
                rows = list(reader)
                self.assertEqual(len(rows), cfg["steps"] + 1)
                first_row = [float(val) for val in rows[0]]
                self.assertEqual(first_row[0], 0)
                self.assertAlmostEqual(first_row[1], 2.8)
                self.assertAlmostEqual(first_row[2], 1.0)

    def test_export_trajectories_byte_reproducible_lf(self):
        """Exported trajectory CSVs must strictly use LF line endings for byte reproducibility."""
        import tempfile

        from scripts.export_trajectories import TRAJECTORY_CONFIGS, export_trajectory

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            cfg = TRAJECTORY_CONFIGS["gd"]
            meta = export_trajectory("gd", cfg, "ill-conditioned", (2.8, 1.0), tmppath)
            csv_path = tmppath / meta["file"]

            content_bytes = csv_path.read_bytes()
            self.assertNotIn(b"\r", content_bytes, "CSV file should use LF line endings, not CRLF")


if __name__ == "__main__":
    unittest.main()
