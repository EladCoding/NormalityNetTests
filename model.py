from test_funcs import *
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, BatchNormalization
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self, output_dim):
        super(MyModel, self).__init__()
        self.dense1 = Dense(128, activation=LeakyReLU(alpha=0.1))
        self.dense2 = Dense(128, activation=LeakyReLU(alpha=0.1))
        self.dense3 = Dense(128, activation=LeakyReLU(alpha=0.1))
        self.dense4 = Dense(output_dim)

    def call(self, x):
        # return x
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)


def create_model(output_dim):
    # define model architecture
    model = MyModel(output_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.015)

    # compile model
    if shapiro_wilk_training_loss:
        training_loss_object = one_dim_shapiro_wilk_loss
        testing_loss_object = one_dim_shapiro_wilk_loss
    else:
        if net_type == 'gaus':
            if random_test_funcs:
                if mean_std_training_loss:
                    training_loss_object = gaus_mean_std_loss
                else:
                    training_loss_object = normal_distributed_gaus_moments_loss
                testing_loss_object = normal_distributed_gaus_moments_loss
            else:
                training_loss_object = gaus_moments_loss
                testing_loss_object = gaus_moments_loss
        elif net_type == 'fourier':
            if random_test_funcs:
                if mean_std_training_loss:
                    training_loss_object = fourier_mean_std_loss
                else:
                    training_loss_object = random_fourier_moments_loss
                testing_loss_object = random_fourier_moments_loss
            else:
                training_loss_object = fourier_moments_loss
                testing_loss_object = fourier_moments_loss
        else:
            print("No such net type")
            exit(1)

        shapiro_wilk_loss_object = multi_dim_shapiro_wilk_loss
        mean_loss_object = mean_loss
        std_loss_object = std_loss
        kurtosis_loss_object = kurtosis_loss
        cube_loss_object = cube_loss
        ball_loss_object = ball_loss
        gaus_loss_object = gaus_loss

    model.compile(optimizer='adam', loss=training_loss_object)

    return model, optimizer, training_loss_object, testing_loss_object, shapiro_wilk_loss_object,\
           mean_loss_object, std_loss_object, kurtosis_loss_object, cube_loss_object, ball_loss_object, gaus_loss_object
