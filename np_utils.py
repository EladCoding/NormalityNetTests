import matplotlib.pyplot as plt
import numpy as np
from playground import *


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
            plt.plot(curve_x, curve_y, 'o-' + line_color, label=label, markersize=4)
        else:
            plt.errorbar(curve_x, curve_y, std, fmt='o-'+line_color, label=label, ecolor='red', markersize=4)
        # plt.plot(curve_x, curve_y, 'ro', label=label)
        if axis:
            plt.axis([x_min, x_max, y_min, y_max])
    plt.legend(loc="upper right")
    plt.savefig(title)
    plt.show()
    plt.clf()


def np_calc_gaus_moment(y_pred, mu, sigma):
    x = y_pred - mu
    x = np.power(np.linalg.norm(x, axis=1), 2)
    x = np.divide(x, 2 * (sigma**2))
    x = np.exp(-x)
    return np.mean(x, axis=0)


def np_calc_fourier_moment(sampled_data, omega):
    x = np.inner(sampled_data, omega)
    x = np.cos(x)
    x = np.mean(x, axis=0)
    return x


def np_calc_fourier_moments(sampled_data, omegas):
    moments = []
    for omega in omegas:
        moments.append(np_calc_fourier_moment(sampled_data, omega))
    return moments


def np_calc_gaus_grid_moments(sampled_data, grid, sigma):
    moments = []
    for mu in grid:
        cur_moment = np_calc_gaus_moment(sampled_data, mu, sigma)
        moments.append(cur_moment)
    return moments


def sample_normal_distribution_with_different_kurtosis(alpha, dim, batch_size):
    mean = np.full(shape=dim, fill_value=0.0)
    cov = np.eye(dim, dim)
    normal_data = np.random.multivariate_normal(mean, cov, size=batch_size)
    abs_normal_data = np.abs(normal_data)
    signed_normal_data = normal_data / abs_normal_data
    changed_kurtosis_data = np.multiply(signed_normal_data, np.power(abs_normal_data, alpha))
    return changed_kurtosis_data


def sample_one_dim_mixture_of_normal_distributions(batch_size, dim):
    sampled_matrix = np.random.rand(batch_size, dim)
    normal_spots = sampled_matrix < 0.8
    number_of_normals = np.sum(normal_spots)
    sampled_matrix[normal_spots] = np.random.normal(size=number_of_normals)

    sampled_matrix[~normal_spots] = np.random.normal(loc=-0.3, scale=2.0, size=(batch_size*dim)-number_of_normals)

    return sampled_matrix


def get_uniform_grid(total_points, radius):
    grid_n = int((np.sqrt(total_points))) - 1
    dist_between_points = (radius * 2) / grid_n
    grid_x = np.arange(-radius, radius + dist_between_points, dist_between_points)
    grid_y = np.arange(-radius, radius + dist_between_points, dist_between_points)
    grid = []
    for x in grid_x:
        for y in grid_y:
            grid.append((x, y))
    return grid


def normal_sigma_generator_heuristic(mu, scale_factor):
    sigma = (np.math.pow(np.linalg.norm(mu), 2) * scale_factor) + 0.05
    return sigma


