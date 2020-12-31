from model import *
train_loss_saver = tf.keras.metrics.Mean(name='train_loss')
test_loss_saver = tf.keras.metrics.Mean(name='basic_test_loss')
shapiro_wilk_loss_saver = tf.keras.metrics.Mean(name='shapiro_wilk_loss')
mean_loss_saver = tf.keras.metrics.Mean(name='mean_loss')
std_loss_saver = tf.keras.metrics.Mean(name='std_loss')
kurtosis_loss_saver = tf.keras.metrics.Mean(name='kurtosis_loss')
cube_loss_saver = tf.keras.metrics.Mean(name='cube_loss')
ball_loss_saver = tf.keras.metrics.Mean(name='ball_loss')
gaus_loss_saver = tf.keras.metrics.Mean(name='gaus_loss')
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

    train_loss_saver(loss)


@tf.function
def test_step(images, labels, model, optimizer, testing_loss_object, shapiro_wilk_loss_object, mean_loss_object,
              std_loss_object, kurtosis_loss_object, cube_loss_object, ball_loss_object, gaus_loss_object):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = testing_loss_object(dist=predictions, number_of_funcs=NUMBER_OF_TEST_FUNCS, out_dim=OUTPUT_DIM)
    test_loss_saver(t_loss)
    sw_loss = shapiro_wilk_loss_object(labels[:32], predictions[:32])
    shapiro_wilk_loss_saver(sw_loss)
    m_loss = mean_loss_object(predictions)
    mean_loss_saver(m_loss)
    st_loss = std_loss_object(predictions)
    std_loss_saver(st_loss)
    k_loss = kurtosis_loss_object(predictions)
    kurtosis_loss_saver(k_loss)
    c_loss = cube_loss_object(predictions)
    cube_loss_saver(c_loss)
    b_loss = ball_loss_object(predictions)
    ball_loss_saver(b_loss)
    g_loss = gaus_loss_object(predictions)
    gaus_loss_saver(g_loss)



def train(model, optimizer, training_loss_object, testing_loss_object, shapiro_wilk_loss_object,
          mean_loss_object, std_loss_object, kurtosis_loss_object, cube_loss_object, ball_loss_object, gaus_loss_object, train_ds, test_ds):

    train_loss_log_dir = 'logs/train_loss'
    test_loss_log_dir = 'logs/basic_test_loss'
    shapiro_wilk_loss_log_dir = 'logs/shapiro_wilk_loss'
    mean_loss_log_dir = 'logs/mean_loss'
    std_loss_log_dir = 'logs/std_loss'
    kurtosis_loss_log_dir = 'logs/kurtosis_loss'
    cube_loss_log_dir = 'logs/cube_loss'
    ball_loss_log_dir = 'logs/ball_loss'
    gaus_loss_log_dir = 'logs/gaus_loss'
    train_loss_summary_writer = tf.summary.create_file_writer(train_loss_log_dir)
    test_loss_summary_writer = tf.summary.create_file_writer(test_loss_log_dir)
    shapiro_wilk_loss_summary_writer = tf.summary.create_file_writer(shapiro_wilk_loss_log_dir)
    mean_loss_summary_writer = tf.summary.create_file_writer(mean_loss_log_dir)
    std_loss_summary_writer = tf.summary.create_file_writer(std_loss_log_dir)
    kurtosis_loss_summary_writer = tf.summary.create_file_writer(kurtosis_loss_log_dir)
    cube_loss_summary_writer = tf.summary.create_file_writer(cube_loss_log_dir)
    ball_loss_summary_writer = tf.summary.create_file_writer(ball_loss_log_dir)
    gaus_loss_summary_writer = tf.summary.create_file_writer(gaus_loss_log_dir)

    for epoch in range(EPOCHS):

        cur_counter = total_counter = 0
        for images, labels in train_ds:
            train_step(images, labels, model, optimizer, training_loss_object)
            cur_counter += 1
            if cur_counter == 100:
                total_counter += cur_counter

                for test_images, test_labels in test_ds:
                    test_step(test_images, test_labels, model, optimizer, testing_loss_object, shapiro_wilk_loss_object,
                              mean_loss_object, std_loss_object, kurtosis_loss_object, cube_loss_object,
                              ball_loss_object, gaus_loss_object)

                with train_loss_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss_saver.result(), step=total_counter)
                with test_loss_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss_saver.result(), step=total_counter)
                with shapiro_wilk_loss_summary_writer.as_default():
                    tf.summary.scalar('loss', shapiro_wilk_loss_saver.result(), step=total_counter)
                with mean_loss_summary_writer.as_default():
                    tf.summary.scalar('loss', mean_loss_saver.result(), step=total_counter)
                with std_loss_summary_writer.as_default():
                    tf.summary.scalar('loss', std_loss_saver.result(), step=total_counter)
                with kurtosis_loss_summary_writer.as_default():
                    tf.summary.scalar('loss', kurtosis_loss_saver.result(), step=total_counter)
                with cube_loss_summary_writer.as_default():
                    tf.summary.scalar('loss', cube_loss_saver.result(), step=total_counter)
                with ball_loss_summary_writer.as_default():
                    tf.summary.scalar('loss', ball_loss_saver.result(), step=total_counter)
                with gaus_loss_summary_writer.as_default():
                    tf.summary.scalar('loss', gaus_loss_saver.result(), step=total_counter)

                test_output = model(test_images)
                plt.clf()
                plt.plot(test_output[:320,0], test_output[:320,1], 'o')
                plt.savefig("img_%6d" % total_counter)
                test_mean = tf.reduce_mean(test_output, axis=0)
                test_std = tf.math.reduce_std(test_output, axis=0)

                test_kurtosis = tf.reduce_mean(((test_output - test_mean) / test_std) ** 4, axis=0)

                print(
                    f'Counter {total_counter}, '
                    f'Train Loss: {train_loss_saver.result()}.\n'
                    f'Basic Test Loss: {test_loss_saver.result()}, '
                    f'Shapiro-wilk Loss: {shapiro_wilk_loss_saver.result()}, '
                    f'mean: {test_mean}, '
                    f'std: {test_std}, '
                    f'kurtosis: {test_kurtosis}.'
                    f'cube loss: {cube_loss_saver.result()}.'
                    f'ball loss: {ball_loss_saver.result()}.'
                    f'gaus loss: {gaus_loss_saver.result()}.'
                )

                cur_counter = 0

                # Reset the metrics at the end of each test step
                train_loss_saver.reset_states()
                shapiro_wilk_loss_saver.reset_states()
                test_loss_saver.reset_states()
                mean_loss_saver.reset_states()
                std_loss_saver.reset_states()
                kurtosis_loss_saver.reset_states()
                cube_loss_saver.reset_states()
                ball_loss_saver.reset_states()
                gaus_loss_saver.reset_states()

    return model
