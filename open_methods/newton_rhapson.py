def f(x):
  return 2*x**3 - 2*x - 5


def f_prime(x):
  return 6*x**2 - 2


def newton_raphson(f, f_prime, x0, tol=0.0001):
  x_list = [x0]
  iteration = 1

  print(f" Iteration | {'x_n':^9s} | {'x_n+1':^9s} | {'Difference'}")

  while True:
    x_new = x_list[-1] - f(x_list[-1]) / f_prime(x_list[-1])
    x_list.append(x_new)
    difference = abs(x_new - x_list[-2])

    print(
        f" {iteration:9d} | {x_list[-2]:^9.5f} | {x_new:^9.5f} | {difference:.5f}")

    if difference < tol:
      print("Approximated root: ", x_new)
      break

    iteration += 1


newton_raphson(f, f_prime, 1.5)
