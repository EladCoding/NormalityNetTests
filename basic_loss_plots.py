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


def main():
    normal_data = sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=OUTPUT_DIM, batch_size=NORMAL_POINTS_NUM)
    curves_list = []
    # uniform
    uniform_possible_sigma = np.arange(0.0, 20, 0.2)
    uniform_sigma = []
    uniform_loss = []
    uniform_std = []
    for sigma in uniform_possible_sigma:
        cur_sigma_uniform_loss = []
        if sigma == 0:
            continue
        test_funcs, expected_moments = get_uniform_test_funcs(sigma, normal_data)
        for i in range(RUNNING_REPITITIONS):
            high_kurtosis_data = sample_normal_distribution_with_different_kurtosis(alpha=1.5, dim=OUTPUT_DIM, batch_size=TRAINING_BATCH_SIZE)
            cur_sigma_uniform_loss.append(normalized_np_moments_loss(high_kurtosis_data, test_funcs, expected_moments))
        uniform_sigma.append(sigma)
        mean_sigma_uniform_loss = np.mean(cur_sigma_uniform_loss)
        uniform_loss.append(mean_sigma_uniform_loss)
        std_sigma_uniform_loss = np.std(cur_sigma_uniform_loss)
        uniform_std.append(std_sigma_uniform_loss)
    curves_list.append((uniform_sigma, uniform_loss, uniform_std, "uniform_loss_curve"))

    plot_graph(curves_list, "losses", "sigma", "loss value")


main()
