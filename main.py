from model import *
import matplotlib.pyplot as plt


def load_training_data(input_dim):
    x_train = tf.random.uniform(shape=(100000, input_dim), dtype=tf.float32)
    y_train = tf.constant(0, shape=(100000, 1), dtype=tf.float32)
    return x_train, y_train


def evaluate_model(model, x_train, y_train, output_dim):
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

    if output_dim == 1:
        plt.hist(tf.squeeze(x_train[:1000]).numpy(), bins=10, label="input")
    else:
        input_x = tf.squeeze(x_train[:1000, 0]).numpy()
        input_y = tf.squeeze(x_train[:1000, 1]).numpy()
        plt.scatter(input_x, input_y, label="input")
    plt.show()

    if output_dim == 1:
        plt.hist(tf.squeeze(model(x_train[:1000])).numpy(), bins=10, label="before")
    else:
        before = model(x_train[:1000])
        before_x = tf.squeeze(before[:, 0]).numpy()
        before_y = tf.squeeze(before[:, 1]).numpy()
        plt.scatter(before_x, before_y, label="before")
    plt.show()

    model.fit(x_train, y_train, batch_size=32, epochs=1)

    if output_dim == 1:
        plt.hist(tf.squeeze(model(x_train[:1000])).numpy(), bins=10, label="after")
    else:
        after = model(x_train[:1000])
        after_x = tf.squeeze(after[:, 0]).numpy()
        after_y = tf.squeeze(after[:, 1]).numpy()
        plt.scatter(after_x, after_y, label="after")
    plt.show()


def main():
    input_dim = output_dim = 2
    x_train, y_train = load_training_data(input_dim)
    model = create_model(output_dim)
    evaluate_model(model, x_train, y_train, output_dim)


main()
