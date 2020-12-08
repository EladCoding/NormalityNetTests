from utils import *


def sample_normal_distribution_with_different_kurtosis(alpha, dim, batch_size):
    mean = np.full(shape=dim, fill_value=1.0)
    cov = np.eye(dim, dim)
    normal_data = np.random.multivariate_normal(mean, cov, size=(batch_size), check_valid='warn', tol=1e-8)
    abs_normal_data = np.abs(normal_data)
    signed_normal_data = normal_data / abs_normal_data
    changed_kurtosis_data = np.multiply(signed_normal_data, np.float_power(abs_normal_data, alpha))
    return changed_kurtosis_data


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
        possible_sigma = np.arange(0.0, 10, 0.2)
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
        cur_sigma_uniform_loss = []
        if sigma == 0:
            continue
        if grid_type == "uniform":
            test_funcs, expected_moments = get_uniform_test_funcs(sigma, normal_data)
        if grid_type == "normal_heuristic":
            test_funcs, expected_moments = get_normal_heuristics_test_funcs(sigma, normal_data)
        for i in range(RUNNING_REPITITIONS):
            high_kurtosis_data = sample_normal_distribution_with_different_kurtosis(alpha=1.5, dim=OUTPUT_DIM, batch_size=TRAINING_BATCH_SIZE)
            normal_kurtosis_data = sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=OUTPUT_DIM, batch_size=TRAINING_BATCH_SIZE)
            cur_sigma_uniform_loss.append(normalized_np_moments_loss(high_kurtosis_data, test_funcs, expected_moments) - normalized_np_moments_loss(normal_kurtosis_data, test_funcs, expected_moments))
        sigma_list.append(sigma)
        mean_sigma_uniform_loss = np.mean(cur_sigma_uniform_loss)
        loss_list.append(mean_sigma_uniform_loss)
        std_sigma_uniform_loss = np.std(cur_sigma_uniform_loss)
        std_list.append(std_sigma_uniform_loss)

        # loss_list[-1] /= std_sigma_uniform_loss

    return (sigma_list, loss_list, std_list, grid_type + "_loss_curve")


def main():
    print("----------------------------------------Generating Basic loss plot----------------------------------------")
    normal_data = sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=OUTPUT_DIM, batch_size=NORMAL_POINTS_NUM)
    curves_list = []

    print("------------------------------Generating Uniform loss plot------------------------------")
    curves_list.append(generate_grid_loss_curve("uniform", normal_data))

    # print("------------------------------Generating Normal loss plot------------------------------")
    # curves_list.append(generate_grid_loss_curve("normal_heuristic", normal_data))

    # print("------------------------------Generating Normal loss plot------------------------------")
    # curves_list.append(generate_grid_loss_curve("normal_heuristic", normal_data))

    print("------------------------------Drawing Normal loss plot------------------------------")
    plot_graph(curves_list, "losses", "sigma", "loss value")


main()
