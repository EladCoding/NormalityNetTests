import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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
def two_dim_shapiro_wilk_loss(y_true, y_pred):
    x = y_pred[:,0]
    y = y_pred[:,1]
    return 0.5 * one_dim_shapiro_wilk_loss(y_true, x) + \
           0.5 * one_dim_shapiro_wilk_loss(y_true, y)


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
    return 0.99 * two_dim_shapiro_wilk_loss(y_true, y_pred) +\
           0.05 * mardia_test_loss(y_true, y_pred)


@tf.function
def calc_moment(y_pred, mu, sigma):
    x = tf.subtract(y_pred, mu)
    x = tf.reduce_sum(tf.multiply(x, x), axis=1)
    x = tf.divide(x, sigma**2)
    x = tf.exp(-x) * 100
    return tf.reduce_mean(x, axis=0)


def np_calc_moment(y_pred, mu, sigma):
    x = y_pred - mu
    x = np.sum(np.multiply(x, x), axis=1)
    x = np.divide(x, sigma**2)
    x = np.exp(-x) * 100
    return np.mean(x, axis=0)


@tf.function
def moments_loss(y_true, y_pred, my_test_funcs):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss = 0
    for (mu, sigma, moment) in my_test_funcs:
        mu = tf.cast(mu, tf.float32)
        sigma = tf.cast(sigma, tf.float32)
        loss = loss + (tf.math.pow(calc_moment(y_pred, mu, sigma) - moment, 2))
    return loss
