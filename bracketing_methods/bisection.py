
def f(x: int | float) -> int | float:
  return 2*x**3 - 2*x - 5


def bisection(f: function, a: int | float, b: int | float, tol: float = 0.001) -> None:
  if f(a) * f(b) >= 0:
    raise ValueError("f(a) and f(b) must have different signs.")

  prev_mid = None
  iteration = 1

  print(
      f"| Iteration | {'a':^15} | {'b':^15} | {'mid':^15} | {'f(mid)':^15} | abs error")
  print("-" * 103)

  while True:
    mid = (a + b) / 2
    f_mid = f(mid)
    error = abs(mid - prev_mid) if prev_mid is not None else None

    print(
        f"| {iteration:^9d} | {a:^15.5f} | {b:^15.5f} | {mid:^15.5f} | {f_mid:^15.5f} | {error}")

    if error is not None and error < tol:
      print("-" * 103)
      print(f"Approximated root: ", mid,)
      break

    if f(a) * f_mid < 0:
      b = mid
    else:
      a = mid

    prev_mid = mid
    iteration += 1


if __name__ == "__main__":
  bisection(f, 1, 2, 1)
