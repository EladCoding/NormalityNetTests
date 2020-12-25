import tensorflow as tf
import numpy as np
from playground import *


@tf.function
def Gaussianity_loss(dist, number_of_funcs, R = None):

    dims = tf.shape(dist, out_type=tf.int32)
    dims0f = tf.cast(dims[0], tf.float32)
    dims0 = tf.cast(dims[0], tf.int32)
    dims1 = tf.cast(dims[1], tf.int32)

    # dist = tf.random_normal((dims0,dims1))*0.25

    mn = tf.reduce_mean(dist,axis=0, keepdims=True)
    dist = dist - mn

    R = tf.cast(R, tf.float32)
    if R != None:
        dist = tf.matmul(dist,R)

    # random freqs

    # v = tf.random.normal((1,1,number_of_funcs)) * 10
    # v = tf.random.uniform((1,1,number_of_funcs)) * 5
    # v = tf.random.uniform((1, 1, number_of_funcs), minval=FOURIER_MIN_FREQ, maxval=FOURIER_MAX_FREQ)
    v = tf.random.uniform((1, 1, number_of_funcs), minval=-1, maxval=1) * 5

    print(v)
    mv = tf.tile(v,[dims0,1,1])
    print(mv)

    dist = tf.reshape(dist,(dims0,dims1,1))
    print(dist)

    dist = tf.matmul(dist,mv)
    print(dist)
    dist_cos = tf.reduce_mean(tf.cos(dist),axis=0)
    print(dist_cos)
    dist_sin = tf.reduce_mean(tf.sin(dist),axis=0)

    tar = tf.matmul(tf.ones((dims1,1)),tf.reshape(v**2,(1,number_of_funcs)))
    tar = tf.exp(-0.5*tar)

    fac = tf.sqrt(dims0f) * 0.2

    fourier_loss = fac * tf.reduce_mean(tf.square(tar - dist_cos) + tf.square(dist_sin - 0))

    return fourier_loss


# @tf.function
# def rand_rot(dim):
#     A = np.random.randn(dim,dim)
#     R,_ = np.linalg.qr(A)
#     R[0,:] = -R[0,:]
#     return R


@tf.function
def rand_rot():
    theta = tf.random.uniform((1, 1))
    rotation_matrix = tf.stack([tf.cos(theta),
                              -tf.sin(theta),
                               tf.sin(theta),
                               tf.cos(theta)])
    rotation_matrix = tf.reshape(rotation_matrix, (2,2))
    return rotation_matrix
