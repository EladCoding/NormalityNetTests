import tensorflow as tf
import tensorflow_probability as tfp
from np_utils import *
import raanan_funcs

NORMAL_POINTS = tf.random.normal(shape=(NORMAL_POINTS_NUM, OUTPUT_DIM), dtype=tf.float32)

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
    x = tf.subtract(y_pred, mu)
    x = tf.reduce_sum(tf.multiply(x, x), axis=1)
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
def fourier_moments_loss(y_true, y_pred, my_test_funcs):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss = 0
    for (omega, cos_moment) in my_test_funcs:
        omega = tf.cast(omega, tf.float32)
        moment = tf.cast(moment, tf.float32)
        loss += calc_fourier_moment_diff(y_pred, omega, cos_moment)
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
def normal_distributed_moments_loss(y_pred, number_of_funcs, output_dim):
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if mean_std_training_loss:
        loss = mean_loss(y_pred)
        loss += std_loss(y_pred)
        loss = 0.1 * loss
    else:
        loss = 0

    for i in range(number_of_funcs):
        mu = tf.random.normal(shape=(1, output_dim))
        sigma = (tf.math.pow(tf.math.reduce_euclidean_norm(mu), 2) + 0.7) / 9
        loss = loss + (tf.math.pow(calc_gaus_moment(y_pred, mu, sigma) - calc_gaus_moment(NORMAL_POINTS, mu, sigma), 2))
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
    return tf.reduce_sum(tf.square(tf.reduce_mean(dist, axis=0)))


@tf.function
def std_loss(dist):
    return tf.reduce_sum(tf.square(1 - tf.math.reduce_std(dist, axis=0)))


@tf.function
def kurtosis_loss(dist):
    mean = tf.reduce_mean(dist, axis=0)
    std = tf.math.reduce_std(dist, axis=0)
    kurtosis = tf.reduce_mean(((dist - mean) / std) ** 4, axis=0)
    kurtosis_loss = tf.reduce_sum(tf.square(kurtosis - 3))
    return kurtosis_loss


@tf.function
def random_fourier_moments_loss(y_pred, number_of_funcs, output_dim):
    y_pred = tf.cast(y_pred, tf.float32)

    if mean_std_training_loss:
        loss = mean_loss(y_pred)
        loss += std_loss(y_pred)
        loss = 0.1 * loss
    else:
        loss = 0

    loss += raanan_funcs.Gaussianity_loss(y_pred, number_of_funcs, raanan_funcs.rand_rot())
    return loss


    for i in range(number_of_funcs):
        omega = tf.random.uniform((1, 1), minval=FOURIER_MIN_FREQ, maxval=FOURIER_MAX_FREQ)
        # tf.print(omega)
        theta = tf.random.uniform((1, 1))
        omega = rotate(tf.constant([1.0, 0.0]) * omega, theta)
        # tf.print(omega)
        target_cos_moment = calc_fourier_cos_moment(NORMAL_POINTS, omega)
        loss += (tf.square(calc_fourier_moment_diff(y_pred, omega, target_cos_moment)))
    return loss
