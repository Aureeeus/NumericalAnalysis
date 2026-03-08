import numpy as np


def gauss_jacobi(A, b, x0, iterations=25):
  n = len(b)
  x = x0.copy()
  history = [x.copy()]

  for _ in range(iterations):
    x_new = np.zeros_like(x)
    for i in range(n):
      s = sum(A[i][j] * x[j] for j in range(n) if j != i)
      x_new[i] = (b[i] - s) / A[i][i]
    x = x_new
    history.append(x.copy())

  return np.array(history)


if __name__ == "__main__":
  # -------------------------------
  # Test system
  # -------------------------------
  A = np.array([[4, 1], [2, 3]], dtype=float)
  b = np.array([1, 2], dtype=float)
  x0 = np.array([0.0, 0.0])
  iterations = 20

  jacobi_path = gauss_jacobi(A, b, x0, iterations)
  exact = np.linalg.solve(A, b)

  print(jacobi_path)
  print(exact)
