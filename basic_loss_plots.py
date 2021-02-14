from np_utils import *
from playground import *
from tqdm import tqdm


def generate_fourier_loss_curve(grid_type, normal_data, number_of_funcs, different_distribution_func, dim):
    loss_list = []
    std_list = []
    if small_gaussian_test_function_with_real_mean:
        print("Fourier can't calculate small gaussian real mean right now!")
        # exit(1)

    possible_omega_values = np.arange(0, 3.1, 0.1)
    possible_omegas = possible_omega_values

    for omegas in tqdm(possible_omegas):
        expected_moments = np_calc_fourier_moments(normal_data, [omegas])
        cur_omegas_loss = []
        for i in range(RUNNING_REPETITIONS):
            if small_gaussian_test_function_without_real_mean:
                if SMALL_GAUS_SIGMA is None:
                    small_sigma = omegas / 2
                else:
                    small_sigma = SMALL_GAUS_SIGMA
                moment_loss = small_gaussian_loss(None, [omegas], dim, small_sigma, type='fourier')
            else:
                different_distribution_data = different_distribution_func(TRAINING_BATCH_SIZE)
                moment_loss = calc_moments_loss(different_distribution_data, None, [omegas], expected_moments, type='fourier')
            cur_omegas_loss.append(moment_loss)

        mean_omega_loss = np.abs(np.mean(cur_omegas_loss))
        loss_list.append(mean_omega_loss)
        std_omega_loss = np.std(cur_omegas_loss)
        std_list.append(std_omega_loss)

        if divide_by_std:
            if std_omega_loss != 0:
                loss_list[-1] /= std_omega_loss
            else:
                if loss_list[-1] != 0:
                    if small_gaussian_test_function_without_real_mean and omegas == 0:
                        loss_list[-1] = 0
                        continue
                    print("std 0 and loss != 0")
                    exit(1)

    return (possible_omegas, loss_list, std_list, grid_type + "_loss_curve")


def get_moments(sampled_points, grid, sigma, type):
    if type == "fourier":
        moments = np_calc_fourier_moments(sampled_points, grid)
    elif type == "gaus":
        moments = np_calc_gaus_grid_moments(sampled_points, grid, sigma)
    else:
        print("no such type of moment loss")
        exit(1)
    return moments


def calc_moments_loss(sampled_points, sigma, grid, expected_moments, type):
    cur_moments = get_moments(sampled_points, grid, sigma, type)
    if relative_error:
        if kick_zero_divide_by_zero:
            moments_diff = []
            for i in range(len(grid)):
                if expected_moments[i] == 0 and cur_moments[i] == 0:
                    cur_diff = 0
                elif expected_moments[i] == 0:
                    cur_diff = np.inf
                    print("diff is inf")
                    exit(1)
                else:
                    cur_diff = (expected_moments[i] - cur_moments[i]) / expected_moments[i]
                moments_diff.append(cur_diff)
        else:
            moments_diff = [(expected_moments[i] - cur_moments[i]) / expected_moments[i] for i in range(len(grid))]
    else:
        moments_diff = [expected_moments[i] - cur_moments[i] for i in range(len(grid))]
    return np.mean(moments_diff)


def calc_std_by_real_mean(cur_sigma_loss, orig_sigma, mu_list, dist_sigma, dist_mu, dim):
    obj_expectation = np.mean(scipy_calc_gaus_moments_expectations(mu_list, orig_sigma, dist_sigma, dist_mu, dim))
    variance = 0
    for sigma_loss in cur_sigma_loss:
        variance += np.square(sigma_loss - obj_expectation) / len(cur_sigma_loss)
    return np.sqrt(variance)


def small_gaussian_loss(sigma, grid, dim, small_sigma, type):
    small_gaussian_points = np.random.normal(loc=SMALL_GAUS_MU ,scale=small_sigma, size=(TRAINING_BATCH_SIZE, dim))
    cur_moments = get_moments(small_gaussian_points, grid, sigma, type)
    return np.mean(cur_moments)


def generate_guas_loss_curve_by_grid(grid_type, normal_data, number_of_funcs, different_distribution_func, dim):
    if grid_type == uniform_grid_name:
        my_grid = [np.array([0] * dim)]
        # my_grid = get_uniform_grid(number_of_funcs, CRITICAL_NORMAL_PART_RADIUS, dim)
    elif grid_type == normal_distributed_grid_name:
        pass
    else:
        print("No such grid")
        exit(1)
    possible_sigmas = np.arange(0.1, 5.0, 0.1)

    loss_list = []
    std_list = []

    for sigma in tqdm(possible_sigmas):

        cur_sigma_loss = []
        if grid_type == uniform_grid_name:
            if not small_gaussian_test_function_with_real_mean:
                if real_expected_value:
                    expected_moments = scipy_calc_gaus_moments_expectations(my_grid, sigma)
                else:
                    expected_moments = np_calc_gaus_grid_moments(normal_data, my_grid, sigma)

        for i in range(RUNNING_REPETITIONS):
            if grid_type == normal_distributed_grid_name:
                my_grid = np.random.normal(size=(number_of_funcs, dim))
                if not small_gaussian_test_function_with_real_mean and not small_gaussian_test_function_without_real_mean:
                    if real_expected_value:
                        expected_moments = scipy_calc_gaus_moments_expectations(my_grid, sigma)
                    else:
                        expected_moments = np_calc_gaus_grid_moments(normal_data, my_grid, sigma)

            if small_gaussian_test_function_with_real_mean or small_gaussian_test_function_without_real_mean:
                if SMALL_GAUS_SIGMA is None:
                    small_sigma = sigma / 2
                else:
                    small_sigma = SMALL_GAUS_SIGMA
                moment_loss = small_gaussian_loss(sigma, my_grid, dim, small_sigma, type='gaus')
            else:
                different_distribution_data = different_distribution_func(TRAINING_BATCH_SIZE)
                moment_loss = calc_moments_loss(different_distribution_data, sigma,
                                                        my_grid, expected_moments, type='gaus')
            cur_sigma_loss.append(moment_loss)

        mean_sigma_loss = np.abs(np.mean(cur_sigma_loss))
        loss_list.append(mean_sigma_loss)
        if small_gaussian_test_function_with_real_mean:
            std_sigma_loss = calc_std_by_real_mean(cur_sigma_loss, sigma, my_grid, dist_sigma=small_sigma, dist_mu=SMALL_GAUS_MU, dim=dim)
        else:
            std_sigma_loss = np.std(cur_sigma_loss)

        std_list.append(std_sigma_loss)

        if divide_by_std:
            loss_list[-1] /= std_sigma_loss

    return (possible_sigmas, loss_list, std_list, grid_type + "_loss_curve")


def poisson_value(x, y, C):
    return C / (C + np.square(np.linalg.norm(x - y)))


def WAE_POISSON_LOSS(Z, Z_tilda, C):
    loss = 0
    loop_len = len(Z) # TODO change this len to make it right
    n = loop_len
    same_normalization = n*(n-1)
    different_normalization = np.square(n)
    for l in range(n):
        for j in range(n):
            if l != j:
                loss += poisson_value(Z[l], Z[j], C) / same_normalization
                loss += poisson_value(Z_tilda[l], Z_tilda[j], C) / same_normalization
            loss -= 2 * (poisson_value(Z[l], Z_tilda[j], C) / different_normalization)
    return loss


def WAE_RBF_LOSS(Z, Z_tilda, sigma):
    loop_len = len(Z) # TODO change this len to make it right
    loss = 0
    for l in range(loop_len):
        loss += np_calc_gaus_moment(np.concatenate([Z[:l], Z[l+1:]]), Z[l], sigma)
        loss += np_calc_gaus_moment(Z_tilda, Z[l], sigma)
    for l in range(loop_len):
        loss += np_calc_gaus_moment(np.concatenate([Z_tilda[:l], Z_tilda[l+1:]]), Z_tilda[l], sigma)
        loss += np_calc_gaus_moment(Z, Z_tilda[l], sigma)
    return loss


def generate_WAE_loss_curve(batch_size, number_of_funcs, different_distribution_func, dim, type):
    print("Notice that in WAE batch_size is also number_of-funcs size")
    if small_gaussian_test_function_with_real_mean:
        print("WAE can't calculate small gaussian real mean right now!")
        # exit(1)
    # possible_sigmas = [np.sqrt(2.0*dim)]
    possible_sigmas = np.arange(0.1, 5.0, 0.1)

    loss_list = []
    std_list = []

    for sigma in tqdm(possible_sigmas):
        cur_sigma_loss = []
        for i in range(RUNNING_REPETITIONS):
            target_distribution_data = np.random.normal(size=(number_of_funcs, dim))
            if small_gaussian_test_function_without_real_mean:
                different_distribution_data = np.random.normal(loc=SMALL_GAUS_MU, scale=SMALL_GAUS_SIGMA, size=(batch_size, dim))
            else:
                different_distribution_data = different_distribution_func(batch_size)
            if type == "WAE_RBF":
                cur_loss = WAE_RBF_LOSS(Z=target_distribution_data, Z_tilda=different_distribution_data, sigma=sigma)
            elif type == "WAE_POISSON":
                cur_loss = WAE_POISSON_LOSS(Z=target_distribution_data, Z_tilda=different_distribution_data, C=sigma)
            else:
                print("No such type for WAE")
                exit(1)
            cur_sigma_loss.append(cur_loss)

        mean_sigma_loss = np.abs(np.mean(cur_sigma_loss))
        loss_list.append(mean_sigma_loss)
        std_sigma_loss = np.std(cur_sigma_loss)
        std_list.append(std_sigma_loss)
        if divide_by_std:
            loss_list[-1] /= std_sigma_loss

    return (possible_sigmas, loss_list, std_list, type + "_loss_curve")


def main():
    print("-" * 50 + "Generating Basic loss plot" + "-" * 50)
    normal_data = sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=dist_dim,
                                                                     batch_size=NORMAL_POINTS_NUM)
    curves_list = []

    normal_lambda = lambda x: sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=dist_dim, batch_size=x)
    if run_gaus:
        print("-" * 50 + "Generating Gaus loss plots" + "-" * 50)
        if uniform_grid:
            print("-" * 50 + "Generating Uniform Gaus Grid loss plot" + "-" * 50)
            possible_omegas, loss_list, std_list, curve_label = \
                generate_guas_loss_curve_by_grid(uniform_grid_name, normal_data, NUMBER_OF_TEST_FUNCS,
                                                 different_distribution_samples, dist_dim)
            if subtract_normal_results:
                normal_possible_omegas, normal_loss_list, normal_std_list, normal_curve_label = \
                    generate_guas_loss_curve_by_grid(uniform_grid_name, normal_data, NUMBER_OF_TEST_FUNCS,
                                                     normal_lambda, dist_dim)
                loss_list = [loss_list[i] - normal_loss_list[i] for i in range(len(loss_list))]
            curves_list.append((possible_omegas, loss_list, std_list, curve_label))
        if normal_distributed_grid:
            print("-" * 50 + "Generating Normal Distributed Gaus Grid loss plot" + "-" * 50)
            possible_omegas, loss_list, std_list, curve_label = \
                generate_guas_loss_curve_by_grid(normal_distributed_grid_name, normal_data, NUMBER_OF_TEST_FUNCS,
                                                 different_distribution_samples, dist_dim)
            if subtract_normal_results:
                normal_possible_omegas, normal_loss_list, normal_std_list, normal_curve_label = \
                    generate_guas_loss_curve_by_grid(normal_distributed_grid_name, normal_data, NUMBER_OF_TEST_FUNCS,
                                                     normal_lambda, dist_dim)
                loss_list = [loss_list[i] - normal_loss_list[i] for i in range(len(loss_list))]
            curves_list.append((possible_omegas, loss_list, std_list, curve_label))
    if run_fourier:
        print("-" * 50 + "Generating Fourier loss plot" + "-" * 50)
        possible_omegas, loss_list, std_list, curve_label = \
            generate_fourier_loss_curve("fourier", normal_data, NUMBER_OF_TEST_FUNCS, different_distribution_samples, dist_dim)
        if subtract_normal_results:
            normal_possible_omegas, normal_loss_list, normal_std_list, normal_curve_label = \
                generate_fourier_loss_curve("fourier", normal_data, NUMBER_OF_TEST_FUNCS, normal_lambda, dist_dim)
            loss_list = [loss_list[i] - normal_loss_list[i] for i in range(len(loss_list))]
        curves_list.append((possible_omegas, loss_list, std_list, curve_label))
    if run_WAE_RBF:
        print("-" * 50 + "Generating WAE RBF loss plot" + "-" * 50)
        possible_omegas, loss_list, std_list, curve_label = \
            generate_WAE_loss_curve(TRAINING_BATCH_SIZE, NUMBER_OF_TEST_FUNCS, different_distribution_samples, dist_dim, type="WAE_RBF")
        if subtract_normal_results:
            print("Can't sub normal results from WAE")
            exit(1)
        curves_list.append((possible_omegas, loss_list, std_list, curve_label))
    if run_WAE_POISSON:
        print("-" * 50 + "Generating WAE Poisson loss plot" + "-" * 50)
        possible_omegas, loss_list, std_list, curve_label = \
            generate_WAE_loss_curve(TRAINING_BATCH_SIZE, NUMBER_OF_TEST_FUNCS, different_distribution_samples, dist_dim, type="WAE_POISSON")
        if subtract_normal_results:
            print("Can't sub normal results from WAE")
            exit(1)
        curves_list.append((possible_omegas, loss_list, std_list, curve_label))

    print("-" * 50 + "Drawing loss plots" + "-" * 50)
    plot_graph(curves_list, "losses", x_label_name, "loss value")


main()
