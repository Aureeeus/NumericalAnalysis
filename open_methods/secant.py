
def f(x):
  return 2*x**3 - 2*x - 5


def secant(f, x0, x1, tol=0.001):
  x_list = [x0, x1]
  iteration = 1

  print(f" Iteration | {'x_n':^9s} | {'x_n+1':^9s} | {'Difference'}")

  while True:
    x_new = x_list[-1] - f(x_list[-1]) * (x_list[-1] -
                                          x_list[-2]) / (f(x_list[-1]) - f(x_list[-2]))
    x_list.append(x_new)
    difference = abs(x_new - x_list[-2])

    print(
        f" {iteration:9d} | {x_list[-2]:^9.5f} | {x_new:^9.5f} | {difference:.5f}")

    if difference < tol:
      print("Approximated root: ", x_new)
      break

    iteration += 1


secant(f, 1, 2)
