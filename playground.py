import np_utils

# Problem parameters
INPUT_DIM = 2
OUTPUT_DIM = 2


# Optimization parameters
NUMBER_OF_TEST_FUNCS = 100
TRAINING_BATCH_SIZE = 2000
NORMAL_POINTS_NUM = 32000
CRITICAL_NORMAL_PART_RADIUS = 3.0
TESTING_BATCH_SIZE = 2000
TRAINING_BATCHES = 200000
TRAIN_SIZE = TRAINING_BATCH_SIZE * TRAINING_BATCHES
TEST_SIZE = 100 * TESTING_BATCH_SIZE

# Visualization parameters
TEST_PLOT_EXAMPLES_SIZE = 320


# Playground
# tf
# net_type = 'gaus'
net_type = 'fourier'
random_test_funcs = True
mean_std_training_loss = True
shapiro_wilk_training_loss = False
FOURIER_MIN_FREQ = 0.3
FOURIER_MAX_FREQ = 0.8

#np
def different_distribution_samples(TRAINING_BATCH_SIZE):
    # normal
    # return utils.sample_normal_distribution_with_different_kurtosis(alpha=1.0, dim=dist_dim, batch_size=TRAINING_BATCH_SIZE)
    # high kurtosis
    return np_utils.sample_normal_distribution_with_different_kurtosis(alpha=1.5, dim=dist_dim, batch_size=TRAINING_BATCH_SIZE)
    # mixture of gaussian
    # return utils.sample_one_dim_mixture_of_normal_distributions(TRAINING_BATCH_SIZE, dim=dist_dim)


divide_by_std = False
relative_error = False
run_fourier = True
run_gaus = False
subtract_normal_results = False
x_label_name = "omega"
# x_label_name = "sigma"
RUNNING_REPITITIONS = 100
dist_dim = 2
