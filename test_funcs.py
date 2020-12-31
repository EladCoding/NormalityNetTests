from loss_functions import *


def growing_grid(number_of_funcs=NUMBER_OF_TEST_FUNCS):
    test_funcs = []
    grid_n = int((np.sqrt(number_of_funcs//4)))

    pow_list = np.array([1.5**x for x in range(grid_n)], dtype=np.float32)
    grid_list = pow_list * 0.1
    for i in range(grid_n):
        i_sigma = grid_list[i]
        i_mu = i_sigma - 0.1
        for j in range(grid_n):
            j_sigma = grid_list[j]
            j_mu = j_sigma - 0.1
            sigma = max(i_sigma, j_sigma)
            test_funcs.append(((i_mu, j_mu), sigma, np_calc_gaus_moment(NORMAL_POINTS, (i_mu, j_mu), sigma)))
            test_funcs.append(((-i_mu, j_mu), sigma, np_calc_gaus_moment(NORMAL_POINTS, (-i_mu, j_mu), sigma)))
            test_funcs.append(((i_mu, -j_mu), sigma, np_calc_gaus_moment(NORMAL_POINTS, (i_mu, -j_mu), sigma)))
            test_funcs.append(((-i_mu, -j_mu), sigma, np_calc_gaus_moment(NORMAL_POINTS, (-i_mu, -j_mu), sigma)))
    print(len(test_funcs))
    print(test_funcs[-1])
    return test_funcs


def get_uniform_np_test_funcs(sigma, number_of_funcs, dim):
    test_funcs = []
    grid = get_uniform_grid(number_of_funcs, CRITICAL_NORMAL_PART_RADIUS)
    for (x, y) in grid:
        test_funcs.append(((x, y), sigma, np_calc_gaus_moment(NORMAL_POINTS, (x, y), sigma)))
    return test_funcs


def get_fourier_np_test_funcs(number_of_funcs, dim):
    test_funcs = []
    grid_n = int(np.power(number_of_funcs, 1.0 / dim))
    edges = []
    xyz = []
    for i in range(dim):
        xyz.append(np.array([0, 0, 1]))
        edges.append(np.linspace(FOURIER_MIN_FREQ, FOURIER_MAX_FREQ, grid_n))
    xyz = np.vstack(xyz).T
    hist, edges = np.histogramdd(xyz, edges)
    grid = np.vstack(np.meshgrid(*edges)).reshape(len(edges), -1).T

    for omega in grid:
        test_funcs.append((omega, np_calc_fourier_moment(NORMAL_POINTS, omega)))
    return test_funcs


def small_uniform_grid():
    return get_uniform_np_test_funcs(0.1, number_of_funcs=NUMBER_OF_TEST_FUNCS)


def medium_uniform_grid():
    return get_uniform_np_test_funcs(1.1, number_of_funcs=NUMBER_OF_TEST_FUNCS)


def big_uniform_grid():
    return get_uniform_np_test_funcs(10, number_of_funcs=NUMBER_OF_TEST_FUNCS)


def fourier_grid():
    return get_fourier_np_test_funcs(number_of_funcs=NUMBER_OF_TEST_FUNCS, dim=OUTPUT_DIM)
