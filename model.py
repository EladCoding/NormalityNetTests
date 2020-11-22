from test_funcs import *
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, BatchNormalization
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self, output_dim):
        super(MyModel, self).__init__()

        self.dense1 = Dense(64, activation=LeakyReLU(alpha=0.1))
        self.dense2 = Dense(64 * 64, activation=LeakyReLU(alpha=0.1))
        self.dense2 = Dense(64, activation=LeakyReLU(alpha=0.1))
        self.dense3 = Dense(output_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


def create_model(output_dim):
    # define model architecture
    model = MyModel(output_dim)
    optimizer = tf.keras.optimizers.Adam()

    # compile model
    if output_dim == 1:
        training_loss_object = one_dim_shapiro_wilk_loss
        testing_loss_object = one_dim_shapiro_wilk_loss
    elif output_dim == 2:
        training_loss_object = moments_loss
        testing_loss_object = two_dim_shapiro_wilk_loss
    else:
        print("max dim is currently 2")
        exit(1)
    model.compile(optimizer='adam', loss=training_loss_object)

    return model, optimizer, training_loss_object, testing_loss_object
