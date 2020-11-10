from loss_functions import *
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, BatchNormalization
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self, output_dim):
        super(MyModel, self).__init__()

        self.dense1 = Dense(64, activation=LeakyReLU(alpha=0.1))
        self.dense2 = Dense(64 * 64, activation=LeakyReLU(alpha=0.1))
        self.dense3 = Dense(output_dim)
        self.batch_norm1 = BatchNormalization()

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.batch_norm1(x)


def create_model(output_dim):
    # define model architecture
    model = MyModel(output_dim)
    optimizer = tf.keras.optimizers.Adam()

    # compile model
    if output_dim == 1:
        loss_object = one_dim_shapiro_wilk_loss
    elif output_dim == 2:
        loss_object = py_pingouin_loss
    else:
        print("max dim is currently 2")
        exit(1)
    model.compile(optimizer='adam', loss=loss_object)


    return model, optimizer, loss_object
