"""
Newton-Raphson Method for Nonlinear Systems of Equations
=========================================================
Approximates roots of F(x) = 0 where F: R^n -> R^n using the iterative scheme:

    x_{k+1} = x_k - J(x_k)^{-1} * F(x_k)

HOW TO USE
----------
1. Define your system of equations inside `system()` under the
   ── DEFINE YOUR SYSTEM HERE ── section below.

2. (Optional) Define the analytical Jacobian inside `analytical_jacobian()`
   in the same section. Leave it returning None to use automatic finite
   differences instead.

3. Set your initial guess `x0`, iteration limit, and TOLERANCE in the
   ── CONFIGURE AND RUN ── section at the bottom of this file.

Variable convention
-------------------
Your function receives a single NumPy array `v` of length n (one entry per
unknown). Unpack it however you like, for example:

    x, y    = v          # 2-variable system
    x, y, z = v          # 3-variable system
    x = v[0]; y = v[1]   # explicit indexing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
VectorFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]

# ---------------------------------------------------------------------------
# Logging  (change to DEBUG to see extra diagnostics)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column width constants for the table
# ---------------------------------------------------------------------------
COL_ITER = 6    # width of the "Iter" column
COL_VALUE = 18   # width of each x_i and F_i column
COL_RESIDUAL = 20   # width of the "Residual" column
COL_STATUS = 13   # width of the "Status" column


# ---------------------------------------------------------------------------
# Per-iteration record (one row in the table)
# ---------------------------------------------------------------------------
@dataclass
class IterationRecord:
  iteration: int
  x:         NDArray[np.float64]
  fx:        NDArray[np.float64]
  residual:  float
  converged: bool = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NewtonResult:
  """Immutable container for the full solver output."""

  root:            NDArray[np.float64]
  converged:       bool
  iterations_done: int
  residual:        float
  tolerance:       float
  table:           list[IterationRecord] = field(default_factory=list)

  # ------------------------------------------------------------------
  # Pretty-print the iteration table with aligned columns
  # ------------------------------------------------------------------
  def print_table(self, decimals: int = 8) -> None:
    """Print a neatly aligned table of all recorded iterations."""
    if not self.table:
      print("No iteration data recorded.")
      return

    n_vars = len(self.table[0].x)
    n_eqs = len(self.table[0].fx)
    fmt = f".{decimals}f"

    # --- Build header labels ---
    x_labels = [f"x{i + 1}" for i in range(n_vars)]
    fx_labels = [f"F{i + 1}" for i in range(n_eqs)]

    iter_hdr = f"{'Iter':^{COL_ITER}}"
    x_hdrs = "".join(f"{lbl:^{COL_VALUE}}" for lbl in x_labels)
    fx_hdrs = "".join(f"{lbl:^{COL_VALUE}}" for lbl in fx_labels)
    residual_hdr = f"{'Error ||F(x)||':^{COL_RESIDUAL}}"
    status_hdr = f"{'Status':^{COL_STATUS}}"

    header = f"{iter_hdr} {x_hdrs} {fx_hdrs} {residual_hdr} {status_hdr}"
    separator = "─" * len(header)

    print()
    print(
        f"  Tolerance: {self.tolerance:.2e}  (solver stops when Error ||F(x)|| < Tolerance)")
    print(separator)
    print(header)
    print(separator)

    for rec in self.table:
      iter_col = f"{rec.iteration:^{COL_ITER}}"
      x_cols = "".join(f"{v:^{COL_VALUE}{fmt}}" for v in rec.x)
      fx_cols = "".join(f"{v:^{COL_VALUE}{fmt}}" for v in rec.fx)
      residual_col = f"{rec.residual:^{COL_RESIDUAL}.10f}"
      status_col = f"{'✓ Converged':^{COL_STATUS}}" if rec.converged else f"{'':^{COL_STATUS}}"

      print(f"{iter_col} {x_cols} {fx_cols} {residual_col} {status_col}")

    print(separator)

  # ------------------------------------------------------------------
  # Summary
  # ------------------------------------------------------------------
  def __str__(self) -> str:
    status = "CONVERGED" if self.converged else "DID NOT CONVERGE"
    root_str = np.array2string(self.root, precision=10, suppress_small=True)
    return (
        f"\n{'─' * 50}\n"
        f"  Status          : {status}\n"
        f"  Iterations done : {self.iterations_done}\n"
        f"  Tolerance set   : {self.tolerance:.2e}\n"
        f"  Final residual  : {self.residual:.6e}\n"
        f"  Approximate root: {root_str}\n"
        f"{'─' * 50}"
    )


# ---------------------------------------------------------------------------
# Jacobian via central finite differences (automatic fallback)
# ---------------------------------------------------------------------------
def finite_difference_jacobian(
    f:       VectorFunction,
    x:       NDArray[np.float64],
    epsilon: float = 1e-8,
) -> NDArray[np.float64]:
  """
  Approximate the Jacobian of `f` at `x` using central finite differences.

  Parameters
  ----------
  f       : The vector-valued function F: R^n -> R^n.
  x       : Point at which to evaluate the Jacobian.
  epsilon : Finite-difference step size (1e-8 suits most cases).

  Returns
  -------
  J : (n x n) Jacobian matrix.
  """
  n = len(x)
  jac = np.zeros((n, n), dtype=np.float64)
  for i in range(n):
    x_fwd = x.copy()
    x_fwd[i] += epsilon
    x_bwd = x.copy()
    x_bwd[i] -= epsilon
    jac[:, i] = (f(x_fwd) - f(x_bwd)) / (2.0 * epsilon)
  return jac


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------
def newton_raphson(
    f:              VectorFunction,
    x0:             NDArray[np.float64],
    jacobian:       VectorFunction | None = None,
    num_iterations: int = 10,
    tol:            float = 1e-6,
) -> NewtonResult:
  """
  Solve the nonlinear system F(x) = 0 via the Newton-Raphson method.

  The solver runs for at most `num_iterations` steps and stops early as
  soon as the L2-norm of F(x) falls below `tol`.

  Parameters
  ----------
  f              : Vector-valued function F: R^n -> R^n.
  x0             : Initial guess — 1-D array of length n.
  jacobian       : Optional analytical Jacobian J: R^n -> R^(n x n).
                   Falls back to central finite differences if None.
  num_iterations : Maximum number of iterations to perform.
  tol            : Convergence tolerance — stops as soon as
                   ||F(x)|| < tol.

  Returns
  -------
  NewtonResult with root, convergence status, and full iteration table.

  Raises
  ------
  ValueError  : If x0 is not a 1-D array.
  LinAlgError : If the Jacobian becomes singular during iteration.
  """
  x = np.asarray(x0, dtype=np.float64).copy()

  if x.ndim != 1:
    raise ValueError(f"x0 must be a 1-D array, got shape {x.shape}.")

  _jac_fn: VectorFunction = jacobian or (
      lambda xk: finite_difference_jacobian(f, xk)
  )

  table: list[IterationRecord] = []

  for iteration in range(1, num_iterations + 1):
    fx = f(x)
    residual = float(np.linalg.norm(fx))
    converged = residual < tol

    table.append(IterationRecord(
        iteration=iteration,
        x=x.copy(),
        fx=fx.copy(),
        residual=residual,
        converged=converged,
    ))

    if converged:
      logger.info(
          "Tolerance reached at iteration %d  (residual %.4e < tol %.4e)",
          iteration, residual, tol,
      )
      return NewtonResult(
          root=x, converged=True, iterations_done=iteration,
          residual=residual, tolerance=tol, table=table,
      )

    J = _jac_fn(x)

    # Solve J·Δx = −F(x)  (more stable than computing J⁻¹ directly)
    try:
      delta = np.linalg.solve(J, -fx)
    except np.linalg.LinAlgError as exc:
      raise np.linalg.LinAlgError(
          f"Jacobian is singular at iteration {iteration}. "
          "Try a different initial guess."
      ) from exc

    x = x + delta

  # Final residual after the last update (not yet recorded)
  residual = float(np.linalg.norm(f(x)))
  logger.warning(
      "Iteration limit (%d) reached without meeting tolerance %.2e. "
      "Final residual: %.4e",
      num_iterations, tol, residual,
  )
  return NewtonResult(
      root=x, converged=False, iterations_done=num_iterations,
      residual=residual, tolerance=tol, table=table,
  )


# =============================================================================
# ── DEFINE YOUR SYSTEM HERE ──────────────────────────────────────────────────
# =============================================================================

def system(v: NDArray[np.float64]) -> NDArray[np.float64]:
  """
  Write each equation of your system as F_i(v) = 0.

  Steps:
    1. Unpack `v` into your named variables (see examples below).
    2. Write each expression rearranged to equal zero.
    3. Return all expressions as np.array([F1, F2, ...]).

  Unpacking examples:
      x, y    = v      # 2-variable system
      x, y, z = v      # 3-variable system

  ------------------------------------------------------------------
  ↓↓ REPLACE EVERYTHING BELOW THIS LINE WITH YOUR OWN EQUATIONS ↓↓
  ------------------------------------------------------------------
  """
  x, y = v
  return np.array([
      x**2 + y**2 - 4,
      x - y - 1
  ])


def analytical_jacobian(v: NDArray[np.float64]) -> NDArray[np.float64] | None:
  """
  (Optional) Analytical Jacobian of your system.

  Entry [i][j] = partial derivative of equation i w.r.t. variable j.

  Leave `return None` to use automatic finite-difference approximation.

  ------------------------------------------------------------------
  ↓↓ REPLACE THE RETURN BELOW WITH YOUR JACOBIAN, OR KEEP None ↓↓
  ------------------------------------------------------------------
  """
  x, y = v
  return np.array([
      [2*x, 2*y],
      [1, -1]
  ])


# =============================================================================
# ── CONFIGURE AND RUN ────────────────────────────────────────────────────────
# =============================================================================
if __name__ == "__main__":

  # ── Initial guess ─────────────────────────────────────────────────────────
  # One float per unknown, in the same order you unpacked them in system().
  x0 = np.array([1.0, 1.0])   # <- adjust length and values to match your system

  # ── Solver settings ───────────────────────────────────────────────────────
  NUM_ITERATIONS = 10     # <- maximum iterations to run
  TOLERANCE = 0.001   # <- solver stops as soon as ||F(x)|| < TOLERANCE

  # ── Run ───────────────────────────────────────────────────────────────────
  _jac = analytical_jacobian(x0)
  _jac_fn = (lambda v: analytical_jacobian(v)) if _jac is not None else None

  result = newton_raphson(
      f=system,
      x0=x0,
      jacobian=_jac_fn,
      num_iterations=NUM_ITERATIONS,
      tol=TOLERANCE,
  )

  # ── Output ────────────────────────────────────────────────────────────────
  # <- change decimals to show more/fewer decimal places
  result.print_table(decimals=8)
  print(result)
