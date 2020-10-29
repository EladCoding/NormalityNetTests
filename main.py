# Import tf
import tensorflow as tf
import numpy as np
from scipy import stats

# # # Load Data
# mnist = tf.keras.datasets.mnist
# #
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# new load data
x_train = tf.random.uniform(shape=(100000, 1), dtype=tf.float32)
y_train = tf.constant(0, shape=(100000, 1), dtype=tf.float32)

# Build model
model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(input_shape=(1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(1)
])

predictions = model(x_train)
print(predictions[:5])


def helper_pyfunc(y_pred, y_true):
    shapiro_test  = stats.shapiro(y_pred)
    # print(shapiro_test)
    return np.full(fill_value=1 - shapiro_test.statistic, shape=tf.shape(y_true))


def ident(y):
    print("a")
    if True:
        return y


# define loss
old_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# new define loss
# def custom_loss(y_true, y_pred):
#     # breakpoint()
#     # print('a')
#     y = tf.py_function(func=helper_pyfunc, inp=[y_pred, y_true], Tout=tf.float32)
#     # y = tf.py_function(func=ident, inp=[y], Tout=tf.float32)
#     print(y)
#     y = old_loss(y_true, y_pred)
#     print(y)
#     return y


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
def loss_func(y_true, y_pred):
    # print('y_pred')
    # tf.print(y_pred)
    y_pred_sum = tf.reduce_sum(y_pred)
    y_pred_mean = y_pred_sum / n
    # print('y_pred_sum')
    # tf.print(y_pred_sum)
    y_pred_sorted = tf.sort(y_pred, axis=0)
    # print('y_pred_sorted')
    # tf.print(y_pred_sorted)
    y_pred_normalized = tf.math.subtract(y_pred_sorted, y_pred_mean)
    # print('y_pred_normalized')
    # tf.print(y_pred_normalized)
    squared_y_pred_normalized = tf.math.multiply(y_pred_normalized, y_pred_normalized)
    SS = tf.reduce_sum(squared_y_pred_normalized)

    b = 0
    for i in range(m):
        b += a_arr[i] * (y_pred_sorted[n-1-i] - y_pred_sorted[i])

    b_squared = b * b

    return 1 - (b_squared / SS)


# def loss(model, x, y, training):
#   # training=training is needed only if there are layers with different
#   # behavior during training versus inference (e.g. Dropout).
#   y_ = model(x, training=training)
#   return custom_loss(y_true=y, y_pred=y_)
#
#
# l = loss(model, x_train, y_train, training=False)
# print(l)
#
#
# def grad(model, inputs, targets):
#   with tf.GradientTape() as tape:
#     loss_value = loss(model, inputs, targets, training=True)
#   return loss_value, tape.gradient(loss_value, model.trainable_variables)
#
#
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#
#
# loss_value, grads = grad(model, x_train, y_train)
#
# print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
#                                           loss_value.numpy()))
#
# optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
# print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
#                                           loss(model, x_train, y_train, training=True).numpy()))
#







# # Predict
# predictions = model(x_train[:1]).numpy()
# predictions

# # Prob
# tf.nn.softmax(predictions).numpy()

# # check loss
# custom_loss(y_train[:1], predictions).numpy()

# # define model
# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])

# new define model
model.compile(optimizer='adam',
              loss=loss_func)

# fit model
# breakpoint()
# loss_func(None, x_train[:32])
model.fit(x_train, y_train, batch_size=32, epochs=10)


# # Test
# model.evaluate(x_test,  y_test, verbose=2)


