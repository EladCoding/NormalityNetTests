import matplotlib.pyplot as plt
import numpy as np

# Problem parameters
INPUT_DIM = 2
OUTPUT_DIM = 2

# Optimization parameters
NUMBER_OF_TEST_FUNCS = 25
NORMAL_POINTS_NUM = 32000
CRITICAL_NORMAL_PART_RADIUS = 3.0
TRAINING_BATCH_SIZE = 320
TESTING_BATCH_SIZE = 32
TRAINING_BATCHES = 2000
TRAIN_SIZE = TRAINING_BATCH_SIZE * TRAINING_BATCHES
TEST_SIZE = 100 * TESTING_BATCH_SIZE

# Visualization parameters
TEST_PLOT_EXAMPLES_SIZE = 320

# Other parameters
RUNNING_REPITITIONS = 10


def plot_graph(curves_list, title, x_label, y_label, axis=False, x_min=None, x_max=None, y_min=None, y_max=None):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.yscale('log')
    for (curve_x, curve_y, std, label) in curves_list:
        plt.errorbar(curve_x, curve_y, std, fmt='ro', label=label, ecolor='green', markersize=3)
        # plt.plot(curve_x, curve_y, 'ro', label=label, markersize=3)
        # plt.plot(curve_x, curve_y, 'ro', label=label)
        if axis:
            plt.axis([x_min, x_max, y_min, y_max])
    plt.legend(loc="upper right")
    plt.savefig(title)
    plt.show()
    plt.clf()


def np_calc_moment(sampled_data, mu, sigma):
    x = sampled_data - mu
    x = np.sum(np.multiply(x, x), axis=1)
    x = np.divide(x, 2 * (sigma**2))
    x = np.exp(-x)
    return np.mean(x, axis=0)


def normalized_np_moments_loss(sampled_data, test_funcs, expected_moments):
    if len(test_funcs) != len(expected_moments):
        print("len(test_funcs) != len(expected_moments)")
        exit(1)
    loss = 0
    for i in range(len(test_funcs)):
        loss += np.math.pow((np_calc_moment(sampled_data, test_funcs[i][0], test_funcs[i][1]) - expected_moments[i])/expected_moments[i], 2)
        # if loss >= 100:
        #     print("wow")
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
