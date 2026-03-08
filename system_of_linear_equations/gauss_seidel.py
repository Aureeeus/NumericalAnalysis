"""
Gauss-Seidel Iterative Method Solver
=====================================
Solves a 3x3 linear system:  Ax = b
Supports custom equations, initial guesses, and tolerance.
"""

from typing import Callable


# ── Type alias ────────────────────────────────────────────────────────────────
EquationSet = tuple[Callable, Callable, Callable]


# ── Core solver ───────────────────────────────────────────────────────────────
def gauss_seidel(
    equations: EquationSet,
    initial_guess: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tolerance: float = 0.001,
    max_iterations: int = 100,
) -> list[dict]:
  """
  Perform Gauss-Seidel iteration on a 3-variable linear system.

  Parameters
  ----------
  equations : tuple of 3 callables
      Each callable takes (x, y, z) and returns the updated value
      for one variable (rearranged to isolate that variable).
      Example for  4x + y - z = 3  →  x = (3 - y + z) / 4
          lambda x, y, z: (3 - y + z) / 4

  initial_guess : (x0, y0, z0)
      Starting values for the iteration.

  tolerance : float
      Stop when the maximum relative/absolute error falls below this value.

  max_iterations : int
      Safety cap to prevent infinite loops.

  Returns
  -------
  list of dicts containing iteration history.
  """
  x, y, z = initial_guess
  fx, fy, fz = equations
  history: list[dict] = []

  for iteration in range(1, max_iterations + 1):
    x_new = fx(x, y, z)
    y_new = fy(x_new, y, z)
    z_new = fz(x_new, y_new, z)

    # Maximum absolute error across all variables
    error = max(
        abs(x_new - x),
        abs(y_new - y),
        abs(z_new - z),
    )

    history.append(
        {
            "iteration": iteration,
            "x": x_new,
            "y": y_new,
            "z": z_new,
            "error": error,
        }
    )

    x, y, z = x_new, y_new, z_new

    if error < tolerance:
      break

  return history


# ── Pretty-print table ────────────────────────────────────────────────────────
def print_table(history: list[dict], tolerance: float) -> None:
  """Render the iteration results as a formatted table."""
  col_widths = {"iter": 10, "x": 15, "y": 15, "z": 15, "error": 15}

  # Header
  header = (
      f"{'Iter':>{col_widths['iter']}}"
      f"{'x':>{col_widths['x']}}"
      f"{'y':>{col_widths['y']}}"
      f"{'z':>{col_widths['z']}}"
      f"{'Error':>{col_widths['error']}}"
  )
  divider = "─" * len(header)

  print(f"\n{'Gauss–Seidel Iteration Results':^{len(header)}}")
  print(f"{'Tolerance: ' + str(tolerance):^{len(header)}}")
  print(divider)
  print(header)
  print(divider)

  for row in history:
    converged = row["error"] < tolerance
    flag = " ✓" if converged else ""
    print(
        f"{row['iteration']:>{col_widths['iter']}}"
        f"{row['x']:>{col_widths['x']}.6f}"
        f"{row['y']:>{col_widths['y']}.6f}"
        f"{row['z']:>{col_widths['z']}.6f}"
        f"{row['error']:>{col_widths['error']}.6f}"
        f"{flag}"
    )

  print(divider)
  last = history[-1]
  print(f"\n✅ Converged after {last['iteration']} iteration(s).")
  print(f"   x = {last['x']:.6f}")
  print(f"   y = {last['y']:.6f}")
  print(f"   z = {last['z']:.6f}\n")


# ── Interactive input helpers ─────────────────────────────────────────────────
def get_float(prompt: str, default: float | None = None) -> float:
  while True:
    raw = input(prompt).strip()
    if raw == "" and default is not None:
      return default
    try:
      return float(raw)
    except ValueError:
      print("  ⚠  Please enter a valid number.")


def build_equations() -> EquationSet:
  """
  Guide the user to enter a 3×3 system in the form:
      a1*x + b1*y + c1*z = d1
      a2*x + b2*y + c2*z = d2
      a3*x + b3*y + c3*z = d3

  The method automatically rearranges each equation to isolate the
  diagonal variable (Gauss-Seidel requires diagonal dominance).
  """
  print("\n" + "═" * 60)
  print(" Enter your 3×3 system:  a·x + b·y + c·z = d")
  print("═" * 60)

  coeffs = []
  var_names = ["x", "y", "z"]

  for i, var in enumerate(var_names):
    print(f"\n  Equation {i + 1}  (isolates {var}):")
    a = get_float(f"    a{i+1} (coeff of x): ")
    b = get_float(f"    b{i+1} (coeff of y): ")
    c = get_float(f"    c{i+1} (coeff of z): ")
    d = get_float(f"    d{i+1} (RHS value) : ")
    coeffs.append((a, b, c, d))

  # Rearrange each equation to isolate the diagonal variable
  a1, b1, c1, d1 = coeffs[0]  # isolate x  →  x = (d1 - b1*y - c1*z) / a1
  a2, b2, c2, d2 = coeffs[1]  # isolate y  →  y = (d2 - a2*x - c2*z) / b2
  a3, b3, c3, d3 = coeffs[2]  # isolate z  →  z = (d3 - a3*x - b3*y) / c3

  def eq_x(x, y, z): return (d1 - b1 * y - c1 * z) / a1
  def eq_y(x, y, z): return (d2 - a2 * x - c2 * z) / b2
  def eq_z(x, y, z): return (d3 - a3 * x - b3 * y) / c3

  return eq_x, eq_y, eq_z


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
  print("\n╔══════════════════════════════════════════╗")
  print("║   Gauss–Seidel Iterative Method Solver   ║")
  print("╚══════════════════════════════════════════╝")

  use_demo = input(
      "\nUse built-in demo system? (y/n, default=y): "
  ).strip().lower()

  if use_demo in ("", "y"):
    # Demo system (diagonally dominant):
    #   4x +  y -  z =  3
    #   2x + 5y +  z = 15
    #  -x + 2y + 6z = 10
    print("\nDemo system:")
    print("   4x +  y -  z =  3")
    print("   2x + 5y +  z = 15")
    print("  -x + 2y + 6z = 10")

    equations: EquationSet = (
        lambda x, y, z: (7 - y - z) / 4,
        lambda x, y, z: (-8 - x - z) / 5,
        lambda x, y, z: (6 - x - y) / 6,
    )
  else:
    equations = build_equations()

  # ── Initial guess ──────────────────────────────────────────────────────
  print("\n" + "─" * 40)
  print(" Initial Guess  (press Enter for 0.0)")
  print("─" * 40)
  x0 = get_float("  x₀: ", default=0.0)
  y0 = get_float("  y₀: ", default=0.0)
  z0 = get_float("  z₀: ", default=0.0)

  # ── Tolerance ──────────────────────────────────────────────────────────
  tol = get_float("\n Tolerance (default=0.001): ", default=0.001)
  max_iter = int(get_float(" Max iterations  (default=100): ", default=100))

  # ── Solve & display ────────────────────────────────────────────────────
  history = gauss_seidel(
      equations,
      initial_guess=(x0, y0, z0),
      tolerance=tol,
      max_iterations=max_iter,
  )
  print_table(history, tol)


if __name__ == "__main__":
  main()
