import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# Problem parameters
INPUT_DIM = 2
OUTPUT_DIM = 2

# Optimization parameters
NUMBER_OF_TEST_FUNCS = 25
TRAINING_BATCH_SIZE = 100
NORMAL_POINTS_NUM = 32000
CRITICAL_NORMAL_PART_RADIUS = 3.0
TESTING_BATCH_SIZE = 32
TRAINING_BATCHES = 2000
TRAIN_SIZE = TRAINING_BATCH_SIZE * TRAINING_BATCHES
TEST_SIZE = 100 * TESTING_BATCH_SIZE

# Visualization parameters
TEST_PLOT_EXAMPLES_SIZE = 320


def plot_graph(curves_list, title, x_label, y_label, axis=False, x_min=None, x_max=None, y_min=None, y_max=None):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.yscale('log')
    for i, (curve_x, curve_y, std, label) in enumerate(curves_list):
        if i == 0:
            line_color = 'g'
        elif i == 1:
            line_color = 'b'
        elif i == 1:
            line_color = 'y'
        elif i == 1:
            line_color = 'p'
        elif i == 1:
            line_color = 'o'
        else:
            "too many curves"
            exit(1)

        if divide_by_std:
            plt.plot(curve_x, curve_y, 'o-'+line_color, label=label, markersize=4)
        else:
            plt.errorbar(curve_x, curve_y, std, fmt='o-'+line_color, label=label, ecolor='red', markersize=4)
        # plt.plot(curve_x, curve_y, 'ro', label=label)
        if axis:
            plt.axis([x_min, x_max, y_min, y_max])
    plt.legend(loc="upper right")
    plt.savefig(title)
    plt.show()
    plt.clf()


def np_calc_one_dim_moment(sampled_data, mu, sigma):
    x = sampled_data - mu
    x = np.multiply(x, x)
    x = np.divide(x, 2 * (sigma**2))
    x = np.exp(-x)
    return np.mean(x, axis=0)


def np_calc_fourier_moment(sampled_data, omega):
    x = np.cos(sampled_data*omega)
    x = np.sum(x, axis=0)
    return x


def np_calc_one_dim_fourier_moments(sampled_data, omegas):
    moments = []
    for omega in omegas:
        # if omega == 0:
        #     print("Omega = 0")
        #     exit(1)
        moments.append(np_calc_fourier_moment(sampled_data, omega) / len(sampled_data))
    return moments


def np_calc_one_dim_uniform_grid_moments(sampled_data, sigma, grid):
    moments = []
    for mu in grid:
        cur_moment = np_calc_one_dim_moment(sampled_data, mu, sigma)
        moments.append(cur_moment)
    return moments


def sample_normal_distribution_with_different_kurtosis(alpha, dim, batch_size):
    # mean = np.full(shape=dim, fill_value=1.0)
    # cov = np.eye(dim, dim)
    # normal_data = np.random.multivariate_normal(mean, cov, size=(batch_size), check_valid='warn', tol=1e-8)
    normal_data = np.random.normal(size=TRAINING_BATCH_SIZE)
    abs_normal_data = np.abs(normal_data)
    signed_normal_data = normal_data / abs_normal_data
    changed_kurtosis_data = np.multiply(signed_normal_data, np.power(abs_normal_data, alpha))
    return changed_kurtosis_data


def sample_one_dim_mixture_of_normal_distributions(batch_size):
    sampled_matrix = np.random.rand(batch_size, 1)
    normal_spots = sampled_matrix < 0.8
    number_of_normals = np.sum(normal_spots)
    sampled_matrix[normal_spots] = np.random.normal(size=number_of_normals)

    sampled_matrix[~normal_spots] = np.random.normal(loc=-0.3, scale=2.0, size=batch_size-number_of_normals)

    return sampled_matrix


def normalized_np_moments_loss(sampled_data, test_funcs, expected_moments):
    if len(test_funcs) != len(expected_moments):
        print("len(test_funcs) != len(expected_moments)")
        exit(1)
    loss = 0
    for i in range(len(test_funcs)):
        loss += np.math.pow((np_calc_moment(sampled_data, test_funcs[i][0], test_funcs[i][1]) - expected_moments[i])/expected_moments[i], 2)
    return loss


def get_uniform_grid(total_points, radius):
    grid_n = int((np.sqrt(total_points)))
    dist_between_points = (radius * 2) / grid_n
    grid_x = np.arange(-radius, radius, dist_between_points)
    grid_y = np.arange(-radius, radius, dist_between_points)
    grid = []
    for x in grid_x:
        for y in grid_y:
            grid.append((x, y))
    return grid


def normal_sigma_generator_heuristic(mu, scale_factor):
    sigma = (np.math.pow(np.linalg.norm(mu), 2) * scale_factor) + 0.05
    return sigma


## Playground
def different_distribution_samples(TRAINING_BATCH_SIZE):
    # normal
    # return sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=1, batch_size=TRAINING_BATCH_SIZE)
    # high kurtosis
    return sample_normal_distribution_with_different_kurtosis(alpha=2.0, dim=1, batch_size=TRAINING_BATCH_SIZE)
    # mixture of gaussian
    # return sample_one_dim_mixture_of_normal_distributions(TRAINING_BATCH_SIZE)


divide_by_std = True
relative_error = True
run_fourier = True
run_gaus = False
x_label = "omega"
# x_label = "sigma"
RUNNING_REPITITIONS = 1000
