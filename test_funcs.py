import numpy as np

# x_s = np.arange(-0.5, 0.5, 0.1)
# y_s = np.arange(-0.5, 0.5, 0.1)
# for x in x_s:
#     for y in y_s:
#         test_funcs.append(((x, y), 0.1))

# x_s = np.arange(-3, -0.75, 0.25)
# y_s = np.arange(-3, -0.75, 0.25)
# for x in x_s:
#     for y in y_s:
#         test_funcs.append(((x, y), 0.25))
# x_s = np.arange(0.75, 3, 0.25)
# y_s = np.arange(0.75, 3, 0.25)
# for x in x_s:
#     for y in y_s:
#         test_funcs.append(((x, y), 0.25))

# x_s = np.arange(-6, -3.5, 0.5)
# y_s = np.arange(-6, -3.5, 0.5)
# for x in x_s:
#     for y in y_s:
#         test_funcs.append(((x, y), 0.5))
# x_s = np.arange(3.5, 6, 0.5)
# y_s = np.arange(3.5, 6, 0.5)
# for x in x_s:
#     for y in y_s:
#         test_funcs.append(((x, y), 0.5))


def normal_distribution_grid(number_of_funcs=1000, output_dim=2):
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
            x = i*sigma
            y = j*sigma
            test_funcs.append(((x,y), sigma))
            test_funcs.append(((-x,y), sigma))
            test_funcs.append(((x,-y), sigma))
            test_funcs.append(((-x,-y), sigma))
    print(test_funcs[0])
    return test_funcs


def big_uniform_grid():
    return uniform_grid(1)


def small_uniform_grid():
    return uniform_grid(0.1)
