import tensorflow as tf
import tensorflow_probability as tfp
from np_utils import *
import r_funcs
from scipy.stats import norm
import test_funcs

NORMAL_POINTS = tf.random.normal(shape=(NORMAL_POINTS_NUM, OUTPUT_DIM), dtype=tf.float32) # TODO check maybe use np
NP_NORMAL_POINTS = np.random.normal(size=(NORMAL_POINTS_NUM, OUTPUT_DIM))

n = 32
m = n // 2
a1 = 0.4220
a2 = 0.2921
a3 = 0.2475
a4 = 0.2145
a5 = 0.1874
a6 = 0.1641
a7 = 0.1433
a8 = 0.1243
a9 = 0.1066
a10 = 0.0899
a11 = 0.0739
a12 = 0.0585
a13 = 0.0435
a14 = 0.0289
a15 = 0.0144
a16 = 0.0000


a_arr = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16]
a = tf.constant([a1, a2, a3, a4])
half_y_pred_sorted = tf.Variable(np.zeros(m), dtype=tf.float32)


@tf.function
def correlation_loss(x, y):
    corr = tfp.stats.correlation(x, y, sample_axis=0, event_axis=None)
    return tf.abs(corr)


@tf.function
def one_dim_shapiro_wilk_loss(y_true, y_pred):
    y_pred_sum = tf.reduce_sum(y_pred)
    y_pred_mean = y_pred_sum / n
    y_pred_sorted = tf.sort(y_pred, axis=0)
    y_pred_normalized = tf.math.subtract(y_pred_sorted, y_pred_mean)
    squared_y_pred_normalized = tf.math.multiply(y_pred_normalized, y_pred_normalized)
    SS = tf.reduce_sum(squared_y_pred_normalized)

    b = 0.0
    for i in range(m):
        b += a_arr[i] * (y_pred_sorted[n-1-i] - y_pred_sorted[i])

    b_squared = b * b

    return (1 - (b_squared / SS))


@tf.function
def norm_from_normal(y_true, y_pred):
    dist = 0.0
    for i in range(32):
        rand_normal_points = tf.random.normal(shape=(1, 2), dtype=tf.float32)

        diff_matrix = y_pred - rand_normal_points
        distance_matrix = tf.norm(diff_matrix, axis=-1)

        dist += tf.math.reduce_min(distance_matrix)

    return dist / 64


@tf.function
def multi_dim_shapiro_wilk_loss(y_true, y_pred, dim=OUTPUT_DIM):
    loss = 0
    for i in range(dim):
        loss += one_dim_shapiro_wilk_loss(y_true, y_pred[:, i])
    return tf.divide(loss, dim)


def mardia_test_loss(y_true, y_pred):
    y_pred_cov = tfp.stats.covariance(y_pred, y=None, sample_axis=0, event_axis=-1, keepdims=False)
    y_pred_inverse_cov = tf.linalg.inv(y_pred_cov, adjoint=False)

    y_pred_mean = tf.reduce_mean(y_pred, axis=0, keepdims=True)
    y_pred_normalized = y_pred - y_pred_mean

    yp = 0.0
    kp = 0.0
    n = tf.shape(y_pred)[0]

    for i in range(n):
        for j in range(n):
            dij = tf.reduce_sum(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(tf.reshape(y_pred_normalized[i], (2,1))), y_pred_inverse_cov), (tf.reshape(y_pred_normalized[j], (2,1)))))
            yp += tf.math.pow(dij, 3)
            if j == i:
                kp += tf.math.pow(dij, 2)
    yp = tf.divide(yp, tf.cast(tf.math.pow(n, 2), tf.float32))
    kp = tf.divide(kp, float(n))

    return tf.abs(yp) + tf.abs(kp)


def my_multi_loss(y_true, y_pred):
    return 0.99 * multi_dim_shapiro_wilk_loss(y_true, y_pred) +\
           0.05 * mardia_test_loss(y_true, y_pred)


@tf.function
def calc_gaus_moment(y_pred, mu, sigma):
    y_pred = tf.cast(y_pred, tf.float32)
    x = tf.subtract(y_pred, mu)
    x = tf.reduce_sum(tf.square(x), axis=1)
    x = tf.divide(x, 2 * (sigma**2))
    x = tf.exp(-x)
    return tf.reduce_mean(x, axis=0)


@tf.function
def calc_fourier_moment_diff(dist, omega, target_cos_moment):
    dist = tf.multiply(dist, omega)
    dist = tf.reduce_sum(dist, axis=1)
    dist_cos = tf.math.cos(dist)
    dist_cos = tf.reduce_mean(dist_cos, axis=0)
    dist_sin = tf.math.cos(dist)
    dist_sin = tf.reduce_mean(dist_sin, axis=0)
    moment = tf.square(dist_cos - target_cos_moment) + tf.square(dist_sin)
    return moment


@tf.function
def calc_fourier_cos_moment(dist, omega):
    dist = tf.multiply(dist, omega)
    dist = tf.reduce_sum(dist, axis=1)
    dist_cos = tf.math.cos(dist)
    dist_cos = tf.reduce_mean(dist_cos, axis=0)
    return dist_cos


@tf.function
def fourier_moments_loss(out_dim, dist, number_of_funcs):
    step_size = 2.0 * np.pi / CRITICAL_NORMAL_PART_RADIUS
    start = 0
    print(number_of_funcs)
    omegas_list = [start + i * step_size for i in range(number_of_funcs)]
    print(omegas_list)
    omegas = tf.constant(omegas_list, shape=(1, 1, number_of_funcs))
    print(omegas)

    y_pred = tf.cast(dist, tf.float32)

    loss = mean_loss(y_pred)
    loss += std_loss(y_pred)
    loss = 0.1 * loss

    loss += r_funcs.Gaussianity_loss(y_pred, NUMBER_OF_TEST_FUNCS, r_funcs.rand_rot(), v=omegas)
    return loss


@tf.function
def gaus_moments_loss(y_true, y_pred, my_test_funcs):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss = 0
    for (mu, sigma, moment) in my_test_funcs:
        mu = tf.cast(mu, tf.float32)
        sigma = tf.cast(sigma, tf.float32)
        loss += tf.square(calc_gaus_moment(y_pred, mu, sigma) - moment)
    return loss


@tf.function
def normal_distributed_gaus_moments_loss(dist, number_of_funcs, out_dim):
    dist = tf.cast(dist, tf.float32)

    loss = 0

    for i in range(number_of_funcs):
        mu = tf.random.normal(shape=(1, out_dim))
        sigma = (tf.square(tf.math.reduce_euclidean_norm(mu)) + 0.7) / 9
        loss += tf.square(calc_gaus_moment(dist, mu, sigma) - calc_gaus_moment(NORMAL_POINTS, mu, sigma))
    return loss


@tf.function
def gaus_mean_std_loss(y_pred, number_of_funcs, output_dim):
    y_pred = tf.cast(y_pred, tf.float32)

    loss = mean_loss(y_pred)
    loss += std_loss(y_pred)

    loss += normal_distributed_gaus_moments_loss(y_pred, number_of_funcs, output_dim)
    return loss


def rotate(points, theta):
    rotation_matrix = tf.stack([tf.cos(theta),
                              -tf.sin(theta),
                               tf.sin(theta),
                               tf.cos(theta)])
    rotation_matrix = tf.reshape(rotation_matrix, (2,2))
    points = tf.reshape(points, (1,2))
    return tf.matmul(points, rotation_matrix)


@tf.function
def mean_loss(dist):
    return MEAN_STD_FACTOR * tf.reduce_sum(tf.square(tf.reduce_mean(dist, axis=0)))


@tf.function
def std_loss(dist):
    return MEAN_STD_FACTOR * tf.reduce_sum(tf.square(1 - tf.math.reduce_std(dist, axis=0)))


@tf.function
def kurtosis_loss(dist):
    mean = tf.reduce_mean(dist, axis=0)
    std = tf.math.reduce_std(dist, axis=0)
    kurtosis = tf.reduce_mean(((dist - mean) / std) ** 4, axis=0)
    # kurtosis_loss = tf.reduce_sum(tf.square(kurtosis - 3))
    kurtosis_loss = tf.reduce_sum(tf.abs(kurtosis - 3))
    return kurtosis_loss


@tf.function
def cube_loss(dist, radius=None, means=None):
    if radius is None:
        radius = np.sqrt(OUTPUT_DIM) * 0.25
    if means is None:
        means = np.arange(-CRITICAL_NORMAL_PART_RADIUS, CRITICAL_NORMAL_PART_RADIUS + 0.1, 0.1)
    loss = 0
    for mean in means:
        is_inside_1 = dist < mean + radius
        is_inside_2 = dist > mean - radius
        is_inside = tf.math.logical_and(is_inside_1, is_inside_2)
        is_inside = tf.reduce_all(is_inside, axis=1)
        perc_inside = tf.reduce_mean(tf.cast(is_inside, tf.float32))

        target_perc_inside = norm.cdf(mean + radius) - norm.cdf(mean - radius)
        target_perc_inside = np.power(target_perc_inside, OUTPUT_DIM)

        loss += tf.abs(perc_inside - target_perc_inside)
        # loss += tf.square(perc_inside - target_perc_inside)
    return loss / len(means)


@tf.function
def ball_loss(dist, radius=None, means=None):
    if radius is None:
        radius = np.sqrt(OUTPUT_DIM) * 0.25
    if means is None:
        means = np.arange(-CRITICAL_NORMAL_PART_RADIUS, CRITICAL_NORMAL_PART_RADIUS + 0.1, 0.1)
    loss = 0
    for mean in means:
        is_inside = dist - mean
        is_inside = tf.norm(is_inside, axis=1)
        is_inside = is_inside < radius
        perc_inside = tf.reduce_mean(tf.cast(is_inside, tf.float32))

        target_is_inside = NP_NORMAL_POINTS - mean
        target_is_inside = np.linalg.norm(target_is_inside, axis=1)
        target_is_inside = target_is_inside < radius
        target_perc_inside = np.mean(target_is_inside.astype(np.float32))

        loss += tf.abs(perc_inside - target_perc_inside)
        # loss += tf.square(perc_inside - target_perc_inside)
    return loss / len(means)


@tf.function
def gaus_loss(dist, sigma=None, means=None):
    if means is None:
        means = np.arange(-CRITICAL_NORMAL_PART_RADIUS, CRITICAL_NORMAL_PART_RADIUS + 0.1, 0.1)
    if sigma is None:
        sigma = 0.1
    my_test_funcs = []
    for mean in means:
        my_test_funcs.append((mean, sigma, np_calc_gaus_moment(NP_NORMAL_POINTS, mean, sigma).astype(np.float32)))
    return gaus_moments_loss(NORMAL_POINTS, dist, my_test_funcs) / len(means)


@tf.function
def fourier_mean_std_loss(dist, number_of_funcs, out_dim):
    dist = tf.cast(dist, tf.float32)

    loss = mean_loss(dist)
    loss += std_loss(dist)
    loss = 0.1 * loss

    loss += random_fourier_moments_loss(dist, number_of_funcs, out_dim)
    return loss


@tf.function
def random_fourier_moments_loss(dist, number_of_funcs, out_dim):
    dist = tf.cast(dist, tf.float32)
    loss = 0
    for i in range(NUMBER_OF_ROTATIONS):
        loss += r_funcs.Gaussianity_loss(dist, number_of_funcs, r_funcs.rand_rot())
    return loss
