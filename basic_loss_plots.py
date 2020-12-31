import itertools

import seaborn as sns

from np_utils import *
from playground import *


def get_uniform_test_funcs(sigma, normal_data):
    test_funcs = []
    grid = get_uniform_grid(NUMBER_OF_TEST_FUNCS, CRITICAL_NORMAL_PART_RADIUS)
    expected_moments = np.empty(len(grid))
    for i, point in enumerate(grid):
        test_funcs.append((point, sigma))
        normal_expected_moment = np_calc_moment(normal_data, point, sigma)
        expected_moments[i] = normal_expected_moment
    return test_funcs, expected_moments


def get_normal_heuristics_test_funcs(scale_factor, normal_data):
    test_funcs = []
    grid = sample_normal_distribution_with_different_kurtosis(alpha=1.5, dim=OUTPUT_DIM, batch_size=TRAINING_BATCH_SIZE)
    expected_moments = np.empty(len(grid))
    for i, point in enumerate(grid):
        sigma = normal_sigma_generator_heuristic(point, scale_factor)
        test_funcs.append((point, sigma))
        normal_expected_moment = np_calc_moment(normal_data, point, sigma)
        expected_moments[i] = normal_expected_moment
    return test_funcs, expected_moments


def generate_fourier_loss_curve(grid_type, normal_data, number_of_funcs, different_distribution_func, dim):
    loss_list = []
    std_list = []

    possible_omega_values = np.arange(0, 2.1, 0.1)
    # step_size = 0.1
    # start = 0
    # possible_omega_values = [start + i * step_size for i in range(number_of_funcs)]
    # possible_omegas = list(itertools.permutations(possible_omega_values, dim))
    possible_omegas = possible_omega_values
    # possible_omega = range(len(possible_omegas))

    for omegas in possible_omegas:
        print(omegas)
        expected_moments = np_calc_fourier_moments(normal_data, [omegas])
        cur_omegas_loss = []
        for i in range(RUNNING_REPITITIONS):
            different_distribution_data = different_distribution_func(TRAINING_BATCH_SIZE)
            cur_omegas_loss.append(calc_moments_loss(different_distribution_data, None, [omegas], expected_moments, type='fourier'))

        mean_omega_loss = np.abs(np.mean(cur_omegas_loss))
        loss_list.append(mean_omega_loss)
        std_omega_loss = np.std(cur_omegas_loss)
        std_list.append(std_omega_loss)

        if divide_by_std:
            if std_omega_loss != 0:
                loss_list[-1] /= std_omega_loss
            else:
                if loss_list[-1] != 0:
                    print("std 0 and loss != 0")
                    exit(1)

        # print(std_omega_loss)

    return (possible_omegas, loss_list, std_list, grid_type + "_loss_curve")
    # return (possible_omegas, loss_list, std_list, grid_type + "_loss_curve")


def calc_moments_loss(sampled_points, sigma, grid, expected_moments, type):
    if type == "fourier":
        cur_moments = np_calc_fourier_moments(sampled_points, grid)
    elif type == "gaus":
        cur_moments = np_calc_gaus_grid_moments(sampled_points, grid, sigma)
    else:
        print("no such type of moment loss")
        exit(1)
    if relative_error:
        moments_diff = [(expected_moments[i] - cur_moments[i])/expected_moments[i] for i in range(len(grid))]
    else:
        moments_diff = [expected_moments[i] - cur_moments[i] for i in range(len(grid))]
    return np.sum(moments_diff)


def generate_uniform_loss_curve(grid_type, normal_data, number_of_funcs, different_distribution_func, dim):
    uniform_grid = [[0]*dim]
    possible_sigmas = np.arange(0.2, 5.0, 0.1)

    loss_list = []
    std_list = []

    for sigma in possible_sigmas:
        # print(sigma)
        cur_sigma_loss = []
        expected_moments = np_calc_gaus_grid_moments(normal_data, uniform_grid, sigma)

        for i in range(RUNNING_REPITITIONS):
            different_distribution_data = different_distribution_func(TRAINING_BATCH_SIZE)
            cur_sigma_loss.append(calc_moments_loss(different_distribution_data, sigma, uniform_grid, expected_moments, type='gaus'))

        mean_sigma_loss = np.abs(np.mean(cur_sigma_loss))
        loss_list.append(mean_sigma_loss)
        std_sigma_loss = np.std(cur_sigma_loss)
        std_list.append(std_sigma_loss)

        if divide_by_std:
            loss_list[-1] /= std_sigma_loss

    return (possible_sigmas, loss_list, std_list, grid_type + "_loss_curve")


def main():
    print("----------------------------------------Generating Basic loss plot----------------------------------------")
    normal_data = sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=dist_dim, batch_size=NORMAL_POINTS_NUM)
    curves_list = []

    normal_lambda = lambda x: sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=dist_dim, batch_size=x)
    if run_gaus:
        print("------------------------------Generating Uniform loss plot------------------------------")
        possible_omegas, loss_list, std_list, curve_label = generate_uniform_loss_curve("gaus", normal_data, NUMBER_OF_TEST_FUNCS, different_distribution_samples, dist_dim)
        if subtract_normal_results:
            normal_possible_omegas, normal_loss_list, normal_std_list, normal_curve_label = generate_uniform_loss_curve("gaus", normal_data, NUMBER_OF_TEST_FUNCS, normal_lambda, dist_dim)
            loss_list = [loss_list[i] - normal_loss_list[i] for i in range(len(loss_list))]
        curves_list.append((possible_omegas, loss_list, std_list, curve_label))
    if run_fourier:
        print("------------------------------Generating Fourier loss plot------------------------------")
        possible_omegas, loss_list, std_list, curve_label = generate_fourier_loss_curve("fourier", normal_data, NUMBER_OF_TEST_FUNCS, different_distribution_samples, dist_dim)
        if subtract_normal_results:
            normal_possible_omegas, normal_loss_list, normal_std_list, normal_curve_label = generate_fourier_loss_curve("fourier", normal_data, NUMBER_OF_TEST_FUNCS, normal_lambda, dist_dim)
            loss_list = [loss_list[i] - normal_loss_list[i] for i in range(len(loss_list))]
        curves_list.append((possible_omegas, loss_list, std_list, curve_label))
        # edge_size = int(np.sqrt(len(possible_omegas)))
        # sns_list = np.zeros(shape=(edge_size, edge_size))
        # counter = 0
        # for i in range(edge_size):
        #     for j in range(edge_size):
        #         sns_list[i,j] = loss_list[counter]
        #         counter += 1
        # ax = sns.heatmap(sns_list, annot=True)
        # plt.show()


    print("------------------------------Drawing Normal loss plot------------------------------")
    plot_graph(curves_list, "losses", x_label_name, "loss value")


main()
