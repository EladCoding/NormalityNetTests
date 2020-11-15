from loss_functions import *

normal_points = tf.random.normal(shape=(32000, 2), dtype=tf.float32)


def normal_distribution_grid(number_of_funcs=1000, output_dim=2):
    test_funcs = []
    mu_list = np.random.normal(size=(number_of_funcs, output_dim)).astype(np.float32)
    for mu in mu_list:
        sigma = 0.1 + np.sqrt(np.linalg.norm(mu))
        moment = np_calc_moment(normal_points, mu, sigma)
        test_funcs.append((mu, sigma, moment))
    print(test_funcs[0])
    return test_funcs


def growing_grid(number_of_funcs=256):
    test_funcs = []
    grid_n = int((np.sqrt(number_of_funcs//4)))

    pow_list = [2**x for x in range(grid_n)]
    for i in range(grid_n):
        i_sigma = pow_list[i]*0.1
        i_mu = i_sigma - 0.1
        for j in range(grid_n):
            j_sigma = pow_list[j]*0.1
            j_mu = j_sigma - 0.1
            sigma = max(i_sigma, j_sigma)
            test_funcs.append(((i_mu,j_mu), sigma, np_calc_moment(normal_points, (i_mu,j_mu), sigma)))
            test_funcs.append(((-i_mu,j_mu), sigma, np_calc_moment(normal_points, (-i_mu,j_mu), sigma)))
            test_funcs.append(((i_mu,-j_mu), sigma, np_calc_moment(normal_points, (i_mu,-j_mu), sigma)))
            test_funcs.append(((-i_mu,-j_mu), sigma, np_calc_moment(normal_points, (-i_mu,-j_mu), sigma)))
    print(test_funcs[0])
    return test_funcs


def uniform_grid(sigma, number_of_funcs=1000):
    test_funcs = []
    grid_n = int((np.sqrt(number_of_funcs//4)))
    for i in range(grid_n):
        for j in range(grid_n):
            x = i*float(sigma)
            y = j*float(sigma)
            test_funcs.append(((x,y), sigma, np_calc_moment(normal_points, (x,y), sigma)))
            test_funcs.append(((-x,y), sigma, np_calc_moment(normal_points, (-x,y), sigma)))
            test_funcs.append(((x,-y), sigma, np_calc_moment(normal_points, (x,-y), sigma)))
            test_funcs.append(((-x,-y), sigma, np_calc_moment(normal_points, (-x,-y), sigma)))
    print(test_funcs[0])
    return test_funcs


def big_uniform_grid():
    return uniform_grid(1, number_of_funcs=400)


def small_uniform_grid():
    return uniform_grid(0.1, number_of_funcs=1600)
