from train import *
import matplotlib.pyplot as plt
from datetime import datetime

TRAINING_BATCH_SIZE = 3200
TESTING_BATCH_SIZE = 32
TRAINING_BATCHES = 5000
TRAIN_SIZE = TRAINING_BATCH_SIZE * TRAINING_BATCHES
TEST_SIZE = 20 * TESTING_BATCH_SIZE


def load_training_data(input_dim, output_dim):
    x_train = tf.random.uniform(shape=(TRAIN_SIZE, input_dim), dtype=tf.float32)
    x_test = tf.random.uniform(shape=(TEST_SIZE, input_dim), dtype=tf.float32)
    y_train = tf.random.normal(shape=(TRAIN_SIZE, output_dim), dtype=tf.float32)
    y_test = tf.random.normal(shape=(TEST_SIZE, output_dim), dtype=tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(TRAINING_BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(TESTING_BATCH_SIZE)

    return train_ds, test_ds, x_test, x_train


def evaluate_model(model, optimizer, training_loss_object, testing_loss_object, train_ds, test_ds, output_dim,
                   x_test, x_train, loss_function_name):
    if output_dim not in [1, 2]:
        print("Unvalid output dim")
        exit(1)
    print(loss_function_name)

    if output_dim == 1:
        plt.hist(tf.squeeze(tf.random.normal(shape=(TEST_SIZE, 1))).numpy(), bins=10, label="normal")
    else:
        normal_x = tf.squeeze(tf.random.normal(shape=(TEST_SIZE, 1))).numpy()
        normal_y = tf.squeeze(tf.random.normal(shape=(TEST_SIZE, 1))).numpy()
        plt.scatter(normal_x, normal_y, label="normal")
    plt.show()

    model, training_loss_list, testing_loss_list = train(model, optimizer, training_loss_object,
                                                         testing_loss_object, train_ds, test_ds)

    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S_")
    training_loss_list = np.array(training_loss_list, dtype=np.float32)
    training_loss_list = training_loss_list / max(training_loss_list)
    testing_loss_list = np.array(testing_loss_list, dtype=np.float32)
    testing_loss_list = testing_loss_list * 10

    plt.clf()
    fig = plt.figure()
    title = loss_function_name
    plt.title(title)
    plt.plot(training_loss_list, label="training loss")
    plt.plot(testing_loss_list, label="shapiro-wilk test loss")
    plt.legend(loc="upper right")
    fig.savefig(date_time+title)
    fig.show()
    plt.clf()

    if output_dim == 1:
        plt.hist(tf.squeeze(model(x_test)).numpy(), bins=10, label="after")
    else:
        after = model(x_test)
        after_x = tf.squeeze(after[:, 0]).numpy()
        after_y = tf.squeeze(after[:, 1]).numpy()
        plt.scatter(after_x, after_y, label="after")
    plt.show()

    if output_dim == 1:
        plt.hist(tf.squeeze(model(x_train[:TEST_SIZE])).numpy(), bins=10, label="after")
    else:
        after = model(x_train[:TEST_SIZE])
        after_x = tf.squeeze(after[:, 0]).numpy()
        after_y = tf.squeeze(after[:, 1]).numpy()
        plt.scatter(after_x, after_y, label="after")
    plt.show()


def main():
    input_dim = output_dim = 2

    train_ds, test_ds, x_test, x_train = load_training_data(input_dim, output_dim)
    model, optimizer, general_training_loss_object, testing_loss_object = create_model(output_dim)
    model(x_train[0:1])
    model.save_weights('model.h5')

    for test_func in [big_uniform_grid, small_uniform_grid, normal_distribution_grid]:
        loss_function_name = test_func.__name__
        my_test_funcs = test_func()
        training_loss_object = lambda y_true, y_pred: general_training_loss_object(y_true, y_pred, my_test_funcs)
        evaluate_model(model, optimizer, training_loss_object, testing_loss_object, train_ds, test_ds, output_dim, x_test, x_train, loss_function_name)
        model.load_weights('model.h5')

main()
