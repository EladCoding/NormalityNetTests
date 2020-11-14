import numpy as np


def normal_distribution_grid(number_of_funcs=14400, output_dim=2):
    test_funcs = []
    mu_list = np.random.normal(size=(number_of_funcs, output_dim)).astype(np.float32)
    for mu in mu_list:
      test_funcs.append((mu, 0.1 + np.sqrt(np.linalg.norm(mu))))
    print(test_funcs[0])
    return test_funcs


def uniform_grid(sigma, number_of_funcs=1000):
    test_funcs = []
    grid_n = int((np.sqrt(number_of_funcs))//4)
    for i in range(grid_n):
        for j in range(grid_n):
            x = i*float(sigma)
            y = j*float(sigma)
            test_funcs.append(((x,y), sigma))
            test_funcs.append(((-x,y), sigma))
            test_funcs.append(((x,-y), sigma))
            test_funcs.append(((-x,-y), sigma))
    print(test_funcs[0])
    return test_funcs


def big_uniform_grid():
    return uniform_grid(1, number_of_funcs=400)


def small_uniform_grid():
    return uniform_grid(0.1, number_of_funcs=14400)
