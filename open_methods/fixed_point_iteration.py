def g(x):
  return ((2*x + 5) / 2)**(1/3)


def fixed_point(g, x0, tol=0.001):
  x_list = [x0]
  iteration = 2

  print(f" Iteration | {'x_n':^9s} | {'x_n+1':^9s} | {'Difference'}")

  while True:
    x_new = g(x_list[-1])
    x_list.append(x_new)
    difference = abs(x_new - x_list[-2])

    print(
        f" {iteration:9d} | {x_list[-2]:^9.5f} | {x_new:^9.5f} | {difference:.5f}")

    if difference < tol:
      print("Approximated root: ", x_new)
      break

    iteration += 1


fixed_point(g, 1.5)
