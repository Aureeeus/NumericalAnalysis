import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import List, Tuple

# =============================================================================
# DATASET — Modify or extend this list freely. Each tuple is (x, y).
# =============================================================================
DATA_POINTS: List[Tuple[float, float]] = [
    (1, 3),
    (2, 4),
    (3, 9),
    (4.6, 18)
]


def round_sympy_expr(expr: sp.Expr, decimals: int = 7) -> sp.Expr:
    """
    Round all floating-point numbers in a SymPy expression to the specified
    number of decimal places.  Whole-number results are promoted to Integer
    so they display without a trailing ".0".
    """
    rounded: sp.Expr = expr.xreplace(
        {n: round(n, decimals) for n in expr.atoms(sp.Number)}
    )
    return rounded.xreplace(
        {n: sp.Integer(int(n)) for n in rounded.atoms(sp.Float) if n == int(n)}
    )


# Symbolic variable used to build the polynomial expression
x = sp.Symbol('x')


def build_lagrange_basis_symbolic(data_points: List[Tuple[float, float]], basis_index: int) -> Tuple[sp.Expr, sp.Expr]:
    """
    Construct the symbolic Lagrange basis polynomial Lₖ(x) for the
    data point at *basis_index*.

    The basis polynomial is defined as:

        L_k(x) = PROD (x - x_j) / (x_k - x_j),  for all j != k

    Returns a *sympy* expression (unsimplified, to preserve the product form
    for display, and a simplified copy).
    """
    x_k = data_points[basis_index][0]
    numerator = sp.Integer(1)
    denominator = sp.Integer(1)

    for j, (x_j, _) in enumerate(data_points):
        if j == basis_index:
            continue
        numerator *= (x - x_j)
        denominator *= (x_k - x_j)

    basis_unsimplified = numerator / denominator
    basis_simplified = sp.simplify(basis_unsimplified)
    return basis_unsimplified, basis_simplified


def build_lagrange_interpolating_polynomial(data_points: List[Tuple[float, float]]) -> sp.Expr:
    """
    Build the full Lagrange interpolating polynomial P(x) symbolically.

        P(x) = Σ yₖ · Lₖ(x),  k = 0 … n

    Returns the expanded and simplified *sympy* expression.
    """
    polynomial = sp.Integer(0)

    for k, (_, y_k) in enumerate(data_points):
        _, basis_simplified = build_lagrange_basis_symbolic(data_points, k)
        polynomial += y_k * basis_simplified

    return sp.expand(polynomial)


def print_basis_table(data_points: List[Tuple[float, float]]) -> List[sp.Expr]:
    """
    Print a detailed table showing, for every data point (xₖ, yₖ):
      • The basis polynomial Lₖ(x) in readable form
      • The weighted term yₖ · Lₖ(x)

    This gives full visibility into each step of the interpolation process.
    """
    total_data_points = len(data_points)

    print("=" * 90)
    print("LAGRANGE POLYNOMIAL INTERPOLATION - PROCESS TABLE")
    print("=" * 90)
    print(f"\nNumber of data points (n + 1): {total_data_points}")
    print(f"Degree of interpolating polynomial: {total_data_points - 1}\n")

    # -- Data-point summary ------------------------------------------------
    print("-" * 50)
    print(f" {'k':^4} | {'x_k':^12} | {'y_k':^12}")
    print("-" * 50)
    for k, (x_k, y_k) in enumerate(data_points):
        print(f" {k:^4} | {x_k:^12.7f} | {y_k:^12.7f}")
    print("-" * 50)

    # -- Basis polynomial details ------------------------------------------
    print("\n" + "=" * 90)
    print("BASIS POLYNOMIALS L_k(x) AND WEIGHTED TERMS y_k * L_k(x)")
    print("=" * 90)

    weighted_terms = []

    for k, (x_k, y_k) in enumerate(data_points):
        unsimplified_basis, simplified_basis = build_lagrange_basis_symbolic(
            data_points, k
        )
        weighted_term = sp.expand(y_k * simplified_basis)
        weighted_terms.append(weighted_term)

        print(f"\n  k = {k}, x_{k} = {x_k:.7f}, y_{k} = {y_k:.7f}")
        print(f"  L_{k}(x) = {round_sympy_expr(simplified_basis, 7)}")
        print(f"  y_{k} · L_{k}(x) = {round_sympy_expr(weighted_term, 7)}")

    print("\n" + "=" * 90)

    return weighted_terms


# Accumulator for points evaluated via evaluate_polynomial(), plotted later
evaluated_points: List[Tuple[float, float]] = []


def print_approximated_function(polynomial_expression: sp.Expr) -> None:
    """
    Neatly display the final approximated interpolating polynomial.
    """
    rounded_polynomial = round_sympy_expr(polynomial_expression, 7)
    print("APPROXIMATED INTERPOLATING POLYNOMIAL P(x)")
    print("=" * 90)
    print(f"\n  P(x) = {rounded_polynomial}\n")
    print("=" * 90)


def     evaluate_polynomial(
    polynomial_expression: sp.Expr,
    input_value: int | float,
) -> float:
    """
    Evaluate the interpolating polynomial at a specific numeric input,
    print the result to the terminal, and store the point so it is
    displayed on the next call to plot_interpolation().

    The output is formatted as an integer when the result has no fractional
    part, otherwise displayed to 7 decimal places.
    """
    result: float = float(polynomial_expression.subs(x, input_value))
    is_whole_number: bool = result == int(result)

    formatted_result: str = str(int(result)) if is_whole_number else f"{result:.7f}"
    print(f"\n  P({input_value}) = {formatted_result}")

    evaluated_points.append((float(input_value), result))

    return result


def plot_interpolation(data_points: List[Tuple[float, float]], polynomial_expression: sp.Expr) -> None:
    """
    Plot the original data points and the interpolating polynomial curve.

    • Data points are drawn as scatter markers in a single colour.
    • The polynomial curve is drawn as a smooth line in a contrasting colour.
    • Any points previously recorded by evaluate_polynomial() are highlighted
      as green diamond markers with coordinate annotations.
    • The plot includes a legend, axis labels, title, and a grid for clarity.
    """
    x_values: List[float] = [point[0] for point in data_points]
    y_values: List[float] = [point[1] for point in data_points]

    # Create a fine range of x values for a smooth curve
    x_minimum: float = min(x_values) - 1
    x_maximum: float = max(x_values) + 1
    x_range_for_curve: np.ndarray = np.linspace(x_minimum, x_maximum, 500)

    # Convert the sympy expression to a callable NumPy function
    polynomial_as_numpy_function = sp.lambdify(x, polynomial_expression, modules='numpy')
    y_range_for_curve: np.ndarray = polynomial_as_numpy_function(x_range_for_curve)

    plt.figure(figsize=(10, 6))

    # Plot the interpolating polynomial curve
    plt.plot(
        x_range_for_curve,
        y_range_for_curve,
        color='#e74c3c',
        linewidth=2,
        label=f'P(x) = {round_sympy_expr(polynomial_expression, 7)}',
    )

    # Scatter-plot the original data points
    plt.scatter(
        x_values,
        y_values,
        color='#2c3e50',
        zorder=5,
        s=80,
        edgecolors='white',
        linewidths=1.2,
        label='Data points',
    )

    # Annotate each data point with its coordinates
    for dx, dy in data_points:
        formatted_dx: str = str(int(dx)) if dx == int(dx) else f"{dx:.7f}"
        formatted_dy: str = str(int(dy)) if dy == int(dy) else f"{dy:.7f}"
        plt.annotate(
            f'({formatted_dx}, {formatted_dy})',
            xy=(dx, dy),
            xytext=(8, -12),
            textcoords='offset points',
            fontsize=9,
            color='#2c3e50',
            fontweight='bold',
        )

    # Highlight any points that were evaluated via evaluate_polynomial()
    if evaluated_points:
        eval_x: List[float] = [p[0] for p in evaluated_points]
        eval_y: List[float] = [p[1] for p in evaluated_points]

        plt.scatter(
            eval_x,
            eval_y,
            color='#27ae60',
            marker='D',
            zorder=6,
            s=80,
            edgecolors='white',
            linewidths=1.2,
            label='Evaluated points',
        )

        for ex, ey in evaluated_points:
            formatted_ey: str = str(int(ey)) if ey == int(ey) else f"{ey:.7f}"
            plt.annotate(
                f'({ex}, {formatted_ey})',
                xy=(ex, ey),
                xytext=(8, -12),
                textcoords='offset points',
                fontsize=9,
                color='#27ae60',
                fontweight='bold',
            )

    plt.title('Lagrange Polynomial Interpolation', fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    # Step 1 — Print the detailed process table for each basis polynomial
    print_basis_table(DATA_POINTS)

    # Step 2 — Build and display the final approximated polynomial
    interpolating_polynomial = build_lagrange_interpolating_polynomial(DATA_POINTS)
    print_approximated_function(interpolating_polynomial)

    # Step 3 — Evaluate P(x) at a specific value (Optional)
    evaluate_polynomial(interpolating_polynomial, 1.6)

    # Step 4 — Graph the dataset alongside the interpolating polynomial
    plot_interpolation(DATA_POINTS, interpolating_polynomial)
