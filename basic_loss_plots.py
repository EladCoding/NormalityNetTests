from utils import *


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


def generate_grid_loss_curve(grid_type, normal_data):
    if grid_type == "uniform":
        possible_sigma = np.arange(0.1, 10, 0.2)
    elif grid_type == "normal_heuristic":
        # Note that here it is a scale factor and not directly sigma
        # possible_sigma = np.arange(0.3, 1.0, 0.1)
        possible_sigma = np.append(np.arange(0.3, 1.0, 0.1), np.arange(3.0, 20.0, 0.5))
    else:
        print("we currently have only uniform and normal heuristics grids")
        exit(1)

    sigma_list = []
    loss_list = []
    std_list = []
    for sigma in possible_sigma:
        print(sigma)
        cur_sigma_loss = []
        if sigma == 0:
            print("Sigma = 0")
            exit(1)
        if grid_type == "uniform":
            test_funcs, expected_moments = get_uniform_test_funcs(sigma, normal_data)
        elif grid_type == "normal_heuristic":
            test_funcs, expected_moments = get_normal_heuristics_test_funcs(sigma, normal_data)
        else:
            print("we currently have only uniform and normal heuristics grids")
            exit(1)
        for i in range(RUNNING_REPITITIONS):
            high_kurtosis_data = sample_normal_distribution_with_different_kurtosis(alpha=1.5, dim=OUTPUT_DIM,
                                                                                    batch_size=TRAINING_BATCH_SIZE)
            # normal_kurtosis_data = sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=OUTPUT_DIM, batch_size=TRAINING_BATCH_SIZE)
            cur_sigma_loss.append(normalized_np_moments_loss(high_kurtosis_data, test_funcs, expected_moments))
        sigma_list.append(sigma)
        mean_sigma_loss = np.mean(cur_sigma_loss)
        loss_list.append(mean_sigma_loss)
        std_sigma_loss = np.std(cur_sigma_loss)
        std_list.append(std_sigma_loss)

        # loss_list[-1] /= std_sigma_uniform_loss

    return (sigma_list, loss_list, std_list, grid_type + "_loss_curve")


def calc_fourier_loss(sampled_points, omegas, expected_moments):
    cur_moments = np_calc_one_dim_fourier_moments(sampled_points, omegas)
    if relative_error:
        moments_diff = [(expected_moments[i] - cur_moments[i])/expected_moments[i] for i in range(len(omegas))]
    else:
        moments_diff = [expected_moments[i] - cur_moments[i] for i in range(len(omegas))]
    return np.sum(moments_diff)


def generate_one_dim_fourier_loss_curve(grid_type, normal_data, number_of_funcs):
    loss_list = []
    std_list = []

    possible_omegas = np.arange(0, 1.1, 0.1)
    # possible_omegas = np.arange(0, 5, 0.1)
    # possible_omegas = [3.0]
    # possible_omegas = [0.5]

    for omega in possible_omegas:
        print(omega)
        expected_moments = np_calc_one_dim_fourier_moments(normal_data, [omega])
        cur_omegas_loss = []
        for i in range(RUNNING_REPITITIONS):
            different_distribution_data = different_distribution_samples(TRAINING_BATCH_SIZE)
            # cur_omegas_loss.append(np_calc_one_dim_fourier_moments(different_distribution_data, [omega]))
            cur_omegas_loss.append(calc_fourier_loss(different_distribution_data, [omega], expected_moments))

        mean_omega_loss = np.abs(np.mean(cur_omegas_loss))
        loss_list.append(mean_omega_loss)
        std_omega_loss = np.std(cur_omegas_loss)
        std_list.append(std_omega_loss)

        if divide_by_std:
            loss_list[-1] /= std_omega_loss

        print(std_omega_loss)

    return (possible_omegas, loss_list, std_list, grid_type + "_loss_curve")


def calc_one_dim_uniform_grid_loss(sampled_points, sigma, uniform_grid, expected_moments):
    cur_moments = np_calc_one_dim_uniform_grid_moments(sampled_points, sigma, uniform_grid)
    if relative_error:
        moments_diff = [(expected_moments[i] - cur_moments[i])/expected_moments[i] for i in range(len(uniform_grid))]
    else:
        moments_diff = [expected_moments[i] - cur_moments[i] for i in range(len(uniform_grid))]
    return np.sum(moments_diff)


def generate_one_dim_uniform_loss_curve(grid_type, normal_data, number_of_funcs):
    # dist_between_sigmas = 6.0 / number_of_funcs
    # uniform_grid = np.arange(-3, 3, dist_between_sigmas)
    # possible_sigmas = np.arange(0.2, 2.0, 0.1)

    # uniform_grid = np.arange(-3, 3, 0.1)
    uniform_grid = [0]
    possible_sigmas = np.arange(0.05, 5.0, 0.1)

    loss_list = []
    std_list = []

    for sigma in possible_sigmas:
        print(sigma)
        cur_sigma_loss = []
        expected_moments = np_calc_one_dim_uniform_grid_moments(normal_data, sigma, uniform_grid)

        for i in range(RUNNING_REPITITIONS):
            different_distribution_data = different_distribution_samples(TRAINING_BATCH_SIZE)
            cur_sigma_loss.append(calc_one_dim_uniform_grid_loss(different_distribution_data, sigma, uniform_grid, expected_moments))

        mean_sigma_loss = np.abs(np.mean(cur_sigma_loss))
        loss_list.append(mean_sigma_loss)
        std_sigma_loss = np.std(cur_sigma_loss)
        std_list.append(std_sigma_loss)

        if divide_by_std:
            loss_list[-1] /= std_sigma_loss
        # print(std_sigma_loss)

    return (possible_sigmas, loss_list, std_list, grid_type + "_loss_curve")


def main():
    print("----------------------------------------Generating Basic loss plot----------------------------------------")
    normal_data = sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=1, batch_size=NORMAL_POINTS_NUM)
    curves_list = []
    number_of_funcs = 10

    # print("------------------------------Generating Uniform loss plot------------------------------")
    # curves_list.append(generate_grid_loss_curve("uniform", normal_data))

    # print("------------------------------Generating Normal loss plot------------------------------")
    # curves_list.append(generate_grid_loss_curve("normal_heuristic", normal_data))

    # print("------------------------------Generating Normal loss plot------------------------------")
    # curves_list.append(generate_grid_loss_curve("normal_heuristic", normal_data))

    if run_gaus:
        print("------------------------------Generating Uniform loss plot------------------------------")
        curves_list.append(generate_one_dim_uniform_loss_curve("gaus", normal_data, number_of_funcs))
    if run_fourier:
        print("------------------------------Generating Fourier loss plot------------------------------")
        curves_list.append(generate_one_dim_fourier_loss_curve("fourier", normal_data, number_of_funcs))

    print("------------------------------Drawing Normal loss plot------------------------------")
    plot_graph(curves_list, "losses", x_label, "loss value")

# for i in range(5):
#     data = different_distribution_samples(TRAINING_BATCH_SIZE)
#     plt.hist(data)
#     plt.show()
# for i in range(5):
#     data = np.random.normal(size=TRAINING_BATCH_SIZE)
#     plt.hist(data)
#     plt.show()

main()
