def f(x):
  return 2*x**3 - 2*x - 5


def regula_falsi(f, a, b, tol):
  if f(a) * f(b) >= 0:
    raise ValueError("f(a) and f(b) must have different signs.")

  iteration = 1
  prev_mid = None

  print(
      f" Iteration | {'a':^9s} | {'b':^9s} | {'mid':^9s} | {'f(mid)':^9s} | abs error")
  print("-" * 70)

  while True:
    mid = (a * f(b) - b * f(a)) / (f(b) - f(a))
    f_mid = f(mid)
    error = abs(mid - prev_mid) if prev_mid is not None else None

    print(
        f" {iteration:^9d} | {a:^9.5f} | {b:^9.5f} | {mid:^9.5f} | {f_mid:^9.5f} | {"None" if error is None else f'{error:.5f}'}")

    if error is not None and error < tol:
      print("-" * 70)
      print(f"Approximated root: ", mid)
      break

    if f(a) * f_mid < 0:
      b = mid
    else:
      a = mid

    prev_mid = mid
    iteration += 1


regula_falsi(f, 1, 2, 0.001)
