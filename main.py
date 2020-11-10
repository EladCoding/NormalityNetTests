from model import *
import matplotlib.pyplot as plt
from train import *


def load_training_data(input_dim):
    x_train = tf.random.uniform(shape=(32 * 3000, input_dim), dtype=tf.float32)
    x_test = tf.random.uniform(shape=(32 * 100, input_dim), dtype=tf.float32)
    y_train = tf.constant(0, shape=(32 * 3000, 1), dtype=tf.float32)
    y_test = tf.constant(0, shape=(32 * 100, 1), dtype=tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    return train_ds, test_ds, x_test


def evaluate_model(model, optimizer, loss_object, train_ds, test_ds, output_dim, x_test):
    if output_dim not in [1, 2]:
        print("Unvalid output dim")
        exit(1)

    if output_dim == 1:
        plt.hist(tf.squeeze(tf.random.normal(shape=(1000, 1))).numpy(), bins=10, label="normal")
    else:
        normal_x = tf.squeeze(tf.random.normal(shape=(1000, 1))).numpy()
        normal_y = tf.squeeze(tf.random.normal(shape=(1000, 1))).numpy()
        plt.scatter(normal_x, normal_y, label="normal")
    plt.show()

    model = train(model, optimizer, loss_object, train_ds, test_ds)

    if output_dim == 1:
        plt.hist(tf.squeeze(model(x_test)).numpy(), bins=10, label="after")
    else:
        after = model(x_test)
        after_x = tf.squeeze(after[:, 0]).numpy()
        after_y = tf.squeeze(after[:, 1]).numpy()
        plt.scatter(after_x, after_y, label="after")
    plt.show()


def main():
    input_dim = output_dim = 2
    train_ds, test_ds, x_test = load_training_data(input_dim)
    model, optimizer, loss_object = create_model(output_dim)
    evaluate_model(model, optimizer, loss_object, train_ds, test_ds, output_dim, x_test)


main()
