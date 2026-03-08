import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import List, Tuple

# =============================================================================
# DATASET — Modify or extend this list freely. Each tuple is (x, y).
# =============================================================================
DATA_POINTS: List[Tuple[float, float]] = [
    (0, 5),
    (2, 7),
    (3, 8),
    (5, 10),
    (6, 12)
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


def build_divided_difference_table(
    data_points: List[Tuple[float, float]],
) -> List[List[float]]:
    """
    Build the full divided-difference table for Newton's interpolation.

    Given n data points, the table is an n × n lower-triangular matrix where:
        table[i][0] = y_i                             (zeroth-order differences)
        table[i][j] = (table[i][j-1] - table[i-1][j-1])
                      / (x_i - x_{i-j})               (j-th order difference)

    Returns the table as a list of lists so every intermediate value can be
    inspected or printed.
    """
    total_points: int = len(data_points)
    table: List[List[float]] = [[0.0] * total_points for _ in range(total_points)]

    # Zeroth-order divided differences are just the y-values
    for i in range(total_points):
        table[i][0] = float(data_points[i][1])

    # Fill higher-order columns
    for column in range(1, total_points):
        for row in range(column, total_points):
            numerator: float = table[row][column - 1] - table[row - 1][column - 1]
            denominator: float = float(
                data_points[row][0] - data_points[row - column][0]
            )
            table[row][column] = numerator / denominator

    return table


def build_newton_interpolating_polynomial(
    data_points: List[Tuple[float, float]],
    divided_difference_table: List[List[float]],
) -> sp.Expr:
    """
    Construct the Newton interpolating polynomial symbolically using the
    divided-difference coefficients (the diagonal of the table).

        P(x) = f[x₀]
             + f[x₀,x₁]·(x - x₀)
             + f[x₀,x₁,x₂]·(x - x₀)(x - x₁)
             + …

    Returns the expanded and simplified *sympy* expression.
    """
    total_points: int = len(data_points)
    polynomial: sp.Expr = sp.Rational(divided_difference_table[0][0]).limit_denominator(10**9)

    accumulated_product: sp.Expr = sp.Integer(1)

    for k in range(1, total_points):
        accumulated_product *= (x - sp.Rational(data_points[k - 1][0]).limit_denominator(10**9))
        coefficient: sp.Rational = sp.Rational(divided_difference_table[k][k]).limit_denominator(10**9)
        polynomial += coefficient * accumulated_product

    return sp.expand(polynomial)


def print_divided_difference_table(
    data_points: List[Tuple[float, float]],
    divided_difference_table: List[List[float]],
) -> None:
    """
    Print the full divided-difference table in a readable format, including:
      • The data-point summary (k, x_k, y_k)
      • Every order of divided differences with 7-decimal formatting

    The layout mirrors the process-table style used in lagrange.py.
    """
    total_points: int = len(data_points)

    print("=" * 90)
    print("NEWTON'S DIVIDED DIFFERENCE INTERPOLATION - PROCESS TABLE")
    print("=" * 90)
    print(f"\nNumber of data points (n + 1): {total_points}")
    print(f"Degree of interpolating polynomial: {total_points - 1}\n")

    # -- Data-point summary ------------------------------------------------
    print("-" * 50)
    print(f" {'k':^4} | {'x_k':^12} | {'y_k':^12}")
    print("-" * 50)
    for k, (x_k, y_k) in enumerate(data_points):
        print(f" {k:^4} | {x_k:^12.7f} | {y_k:^12.7f}")
    print("-" * 50)

    # -- Divided-difference table ------------------------------------------
    print("\n" + "=" * 90)
    print("DIVIDED DIFFERENCE TABLE")
    print("=" * 90)

    # Build the header row dynamically based on the number of data points
    header_parts: List[str] = [f" {'k':^4} | {'x_k':^12} "]
    for order in range(total_points):
        if order == 0:
            header_parts.append(f"{'f[.]':^14}")
        else:
            label: str = f"Order {order}"
            header_parts.append(f"{label:^14}")
    header_line: str = "|".join(header_parts)

    print(header_line)
    print("-" * len(header_line))

    for row in range(total_points):
        x_k: float = float(data_points[row][0])
        row_parts: List[str] = [f" {row:^4} | {x_k:^12.7f} "]

        for column in range(total_points):
            if column <= row:
                row_parts.append(f"{divided_difference_table[row][column]:^14.7f}")
            else:
                row_parts.append(f"{'':^14}")
        print("|".join(row_parts))

    print("=" * 90)

    # -- Diagonal coefficients (the ones used in the polynomial) -----------
    print("\n" + "=" * 90)
    print("DIVIDED DIFFERENCE COEFFICIENTS (DIAGONAL)")
    print("=" * 90)
    for k in range(total_points):
        coefficient: float = divided_difference_table[k][k]
        print(f"  f[x_0, ..., x_{k}] = {coefficient:.7f}")
    print("=" * 90)


# Accumulator for points evaluated via evaluate_polynomial(), plotted later
evaluated_points: List[Tuple[float, float]] = []


def print_approximated_function(polynomial_expression: sp.Expr) -> None:
    """
    Neatly display the final approximated interpolating polynomial.
    """
    rounded_polynomial: sp.Expr = round_sympy_expr(polynomial_expression, 7)
    print("APPROXIMATED INTERPOLATING POLYNOMIAL P(x)")
    print("=" * 90)
    print(f"\n  P(x) = {rounded_polynomial}\n")
    print("=" * 90)


def evaluate_polynomial(
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


def plot_interpolation(
    data_points: List[Tuple[float, float]],
    polynomial_expression: sp.Expr,
) -> None:
    """
    Plot the original data points and the interpolating polynomial curve.

    • Data points are drawn as scatter markers.
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

    plt.title(
        "Newton's Divided Difference Interpolation",
        fontsize=14,
        fontweight='bold',
    )
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
    # Step 1 — Build the divided-difference table
    difference_table: List[List[float]] = build_divided_difference_table(DATA_POINTS)

    # Step 2 — Print the detailed process table
    print_divided_difference_table(DATA_POINTS, difference_table)

    # Step 3 — Build and display the final approximated polynomial
    interpolating_polynomial: sp.Expr = build_newton_interpolating_polynomial(
        DATA_POINTS, difference_table
    )
    print_approximated_function(interpolating_polynomial)

    # Step 4 — Evaluate P(x) at a specific value (Optional)
    evaluate_polynomial(interpolating_polynomial, 4)

    # Step 5 — Graph the dataset alongside the interpolating polynomial
    plot_interpolation(DATA_POINTS, interpolating_polynomial)
