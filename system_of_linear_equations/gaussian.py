
def gaussian_elimination(A, b):
  """
  Solves a system of linear equations Ax = b using Gaussian elimination with partial pivoting.

  Args:
      A: Coefficient matrix (list of lists or numpy array)
      b: Constants vector (list or numpy array)

  Returns:
      x: Solution vector
  """
  n = len(A)

  # Create augmented matrix [A|b]
  augmented = []
  for i in range(n):
    augmented.append(A[i][:] + [b[i]])

  # Forward elimination with partial pivoting
  for col in range(n):
    # Find pivot (largest absolute value in column)
    max_row = col
    for row in range(col + 1, n):
      if abs(augmented[row][col]) > abs(augmented[max_row][col]):
        max_row = row

    # Swap rows
    augmented[col], augmented[max_row] = augmented[max_row], augmented[col]

    # Check for singular matrix
    if abs(augmented[col][col]) < 1e-10:
      raise ValueError("Matrix is singular and cannot be solved")

    # Eliminate below pivot
    for row in range(col + 1, n):
      factor = augmented[row][col] / augmented[col][col]
      for j in range(col, n + 1):
        augmented[row][j] -= factor * augmented[col][j]

  # Back substitution
  x = [0] * n
  for i in range(n - 1, -1, -1):
    x[i] = augmented[i][n]
    for j in range(i + 1, n):
      x[i] -= augmented[i][j] * x[j]
    x[i] /= augmented[i][i]

  return x


if __name__ == "__main__":
  # System of equations:
  # x2 + x3 - 2x4 = -3          (equation 1)
  # x1 + 2x2 - x3 = 2           (equation 2)
  # 2x1 + 4x2 + x3 - 3x4 = -2   (equation 3)
  # x1 - 4x2 - 7x3 - x4 = -19   (equation 4)

  # Coefficient matrix A (reordered for standard form)
  A = [
      [0, 1, 1, -2],        # x2 + x3 - 2x4 = -3
      [1, 2, -1, 0],        # x1 + 2x2 - x3 = 2
      [2, 4, 1, -3],        # 2x1 + 4x2 + x3 - 3x4 = -2
      [1, -4, -7, -1]       # x1 - 4x2 - 7x3 - x4 = -19
  ]

  # Constants vector b
  b = [-3, 2, -2, -19]

  solution = gaussian_elimination(A, b)

  # Display results
  print("Solution:")
  for i, val in enumerate(solution, 1):
    print(f"x{i} = {val:.6f}")

  # Verify solution
  print("\nVerification (Ax = b):")
  for i in range(len(A)):
    result = sum(A[i][j] * solution[j] for j in range(len(A[0])))
    print(f"Equation {i+1}: {result:.6f} = {b[i]}")
