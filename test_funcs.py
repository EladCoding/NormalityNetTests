from loss_functions import *

critical_normal_part = 3.0
NUMBER_OF_FUNCS = 25


def growing_grid(number_of_funcs=NUMBER_OF_FUNCS):
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
            test_funcs.append(((i_mu, j_mu), sigma, np_calc_moment(normal_points, (i_mu, j_mu), sigma)))
            test_funcs.append(((-i_mu, j_mu), sigma, np_calc_moment(normal_points, (-i_mu, j_mu), sigma)))
            test_funcs.append(((i_mu, -j_mu), sigma, np_calc_moment(normal_points, (i_mu, -j_mu), sigma)))
            test_funcs.append(((-i_mu, -j_mu), sigma, np_calc_moment(normal_points, (-i_mu, -j_mu), sigma)))
    print(len(test_funcs))
    print(test_funcs[-1])
    return test_funcs


def uniform_grid(sigma, number_of_funcs):
    test_funcs = []
    grid_n = int((np.sqrt(number_of_funcs)))
    dist_between_points = (critical_normal_part*2)/grid_n
    grid_x = np.arange(-critical_normal_part, critical_normal_part, dist_between_points)
    grid_y = np.arange(-critical_normal_part, critical_normal_part, dist_between_points)
    for x in grid_x:
        for y in grid_y:
            test_funcs.append(((x, y), sigma, np_calc_moment(normal_points, (x, y), sigma)))
    print(len(test_funcs))
    print(test_funcs[-1])
    return test_funcs


def small_uniform_grid():
    return uniform_grid(0.1, number_of_funcs=NUMBER_OF_FUNCS)


def medium_uniform_grid():
    return uniform_grid(1.1, number_of_funcs=NUMBER_OF_FUNCS)


def big_uniform_grid():
    return uniform_grid(10, number_of_funcs=NUMBER_OF_FUNCS)
