from model import *
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
EPOCHS = 1


def random_loss(predictions):
    return np.random.random(predictions.shape[0])


@tf.function
def train_step(images, labels, model, optimizer, training_loss_object):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = training_loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


@tf.function
def test_step(images, labels, model, optimizer, testing_loss_object):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = testing_loss_object(labels, predictions)
    test_loss(t_loss)


def train(model, optimizer, training_loss_object, testing_loss_object, train_ds, test_ds):
    training_loss_list = []
    testing_loss_list = []
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        cur_counter = total_counter = 0
        for images, labels in train_ds:
            train_step(images, labels, model, optimizer, training_loss_object)
            cur_counter += 1
            if cur_counter == 100:
                total_counter += cur_counter

                for test_images, test_labels in test_ds:
                    test_step(test_images, test_labels, model, optimizer, testing_loss_object)

                training_loss_list.append(train_loss.result())
                testing_loss_list.append(test_loss.result())

                test_output = model(test_images)
                test_mean = tf.reduce_mean(test_output, axis=0)
                test_std = tf.math.reduce_std(test_output, axis=0)

                test_kurtosis = tf.reduce_mean(((test_output - test_mean) / test_std) ** 4, axis=0)

                print(
                    f'Counter {total_counter}, '
                    f'Train Loss: {train_loss.result()}, '
                    f'mean: {test_mean}, '
                    f'std: {test_std}, '
                    f'kurtosis: {test_kurtosis}, '
                    f'Shapiro-wilk Loss: {test_loss.result()}'
                )

                cur_counter = 0

    return model, training_loss_list, testing_loss_list
