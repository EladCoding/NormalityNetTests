import np_utils
import numpy as np

# Problem parameters
INPUT_DIM = 2
OUTPUT_DIM = 8


# Optimization parameters
NUMBER_OF_TEST_FUNCS = 100
# NUMBER_OF_ROTATIONS = OUTPUT_DIM
NUMBER_OF_ROTATIONS = 1
TRAINING_BATCH_SIZE = 2000
NORMAL_POINTS_NUM = 320000
CRITICAL_NORMAL_PART_RADIUS = 3.0
TESTING_BATCH_SIZE = 2000
TRAINING_BATCHES = 20000
TRAIN_SIZE = TRAINING_BATCH_SIZE * TRAINING_BATCHES
TEST_SIZE = 100 * TESTING_BATCH_SIZE

# Visualization parameters
TEST_PLOT_EXAMPLES_SIZE = 320


# Playground
# tf
net_type = 'gaus'
# net_type = 'fourier'
# DIR_TITLE = 'gaus_model_128_128_128_lr_0_005_mean_std_freq_normal_dim4'
# DIR_TITLE = 'gaus_model_128_128_128_lr_0_005_mean_std_freq_normal_dim16'
# DIR_TITLE = 'fourier_model_128_128_128_lr_0_005_mean_std_freq_5uniform_dim4'
# DIR_TITLE = 'fourier_model_128_128_128_lr_0_005_freq_5uniform_dim16'
DIR_TITLE = 'gaus_med_sigma_more_losses_mean'
# DIR_TITLE = None
random_test_funcs = True
mean_std_training_loss = True
# sin_1_training_loss = False
# SIN_1_FACTOR = 100
shapiro_wilk_training_loss = False
FOURIER_MIN_FREQ = 0.3
FOURIER_MAX_FREQ = 0.8
MEAN_STD_FACTOR = 0.1


# np
def different_distribution_samples(batch_size):
    # normal
    # return utils.sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=dist_dim, batch_size=batch_size)
    # high kurtosis
    return np_utils.sample_normal_distribution_with_different_kurtosis(alpha=1.5, dim=dist_dim, batch_size=batch_size)
    # Unit cube
    # return np_utils.np.random.uniform(size=[batch_size, dist_dim])
    # mixture of gaussian
    # return np_utils.sample_one_dim_mixture_of_normal_distributions(batch_size, dim=dist_dim)


# General
dist_dim = 8
RUNNING_REPETITIONS = 100
# Only one of these (or none)
std_divided_by_mean_loss = False
mean_sub_by_std_loss = False
median_loss = True
monotonic_loss = False
sigma_monotonic = 0.5
omega_monotonic = 2.15

ignore_negative = True
relative_error = True
kick_zero_divide_by_zero = True
subtract_normal_results = False
small_gaussian_test_function_with_real_mean = False # currently work ok for gaus, and fourier look at gaus with sigma omega/2
small_gaussian_test_function_without_real_mean = False # currently work ok for gaus, and fourier look at gaus with sigma omega/2
SMALL_GAUS_MU = [0.1,0,0,0,0,0,0,0]
SMALL_GAUS_SIGMA = 0.1 # if None, sigma/2 taken
# Fourier
run_fourier = True
# Gaus
run_gaus = True
uniform_grid = True # currently only mean=0
normal_distributed_grid = False
real_expected_value = False # currently dim=1
# WAE RBF
run_WAE_RBF = False
run_WAE_POISSON = False

# x_label_name = "omega"
x_label_name = "sigma"
