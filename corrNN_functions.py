import tensorflow as tf
import numpy as np
import scipy.io

########################################################################################################################


def load_split_dataset(input_file):
    # Load the dataset
    # expecting a .mat file with the variables dataset_XX_input, dataset_XX_ref, t1_t2_combinations, ti_te_combinations
    # for XX being train,val,test, respectively
    data_struct = scipy.io.loadmat(input_file)

    dataset_train = {'input': data_struct['dataset_train_input'], 'ref': data_struct['dataset_train_ref'],
                     'indices': data_struct['dataset_train_indices'], 'no_comp': data_struct['dataset_train_no_comp']}
    dataset_val = {'input': data_struct['dataset_val_input'], 'ref': data_struct['dataset_val_ref'],
                   'indices': data_struct['dataset_val_indices'], 'no_comp': data_struct['dataset_val_no_comp']}
    dataset_test = {'input': data_struct['dataset_test_input'], 'ref': data_struct['dataset_test_ref'],
                    'indices': data_struct['dataset_test_indices'], 'no_comp': data_struct['dataset_test_no_comp']}

    return dataset_train, dataset_val, dataset_test, data_struct['t1_t2_combinations'], data_struct['ti_te_combinations']


def normalize_max_abs(input_array):

    return input_array / np.max(np.abs(input_array))


def add_noise(dataset, percentage_max, random_noise, mu=0, sigma=1):
    dataset['input_noisy'] = np.zeros(dataset['input'].shape)

    if random_noise:
        percentage = np.random.rand(dataset['input'].shape[0], dataset['input'].shape[1]) * percentage_max
    else:
        print('fix noise level')
        percentage = np.ones(dataset['input'].shape[0], dataset['input'].shape[1]) * percentage_max

    # define noise character
    dataset['input_noisy'] = dataset['input'].copy() + percentage / 100. * np.random.normal(mu, sigma, dataset[
        'input'].shape)

    return dataset


def normalize_data(dataset):
    # normalize each of the N signal evolutions to its maximum absolute value
    for jj in range(np.shape(dataset['input_noisy'])[0]):
        dataset['input_noisy'][jj, :] = normalize_max_abs(dataset['input_noisy'][jj, :])
    return dataset


def get_batch(dataset, batch_ind, random_ind_list):
    indexes = random_ind_list[batch_ind]

    input_noisy = dataset['input_noisy'][indexes, :]
    input_clean = dataset['input'][indexes, :]
    ref = dataset['ref'][indexes, :]

    return input_noisy, input_clean, ref


def fc_layer(x, mode, d, n_units, keep, act_fun):
    # Initialize weights and biases
    stddev = tf.cast(tf.sqrt(tf.divide(2, d * n_units)), tf.float32)
    w = tf.Variable(tf.truncated_normal((d, n_units), stddev=stddev, name='weights'))
    b = tf.Variable(tf.zeros([n_units]), name='bias')

    # Multiplication and add
    x_hidden = tf.matmul(x, w)
    x_hidden = tf.add(x_hidden, b)

    # Activation function
    if act_fun == 'softmax':
        x_relu = tf.nn.softmax(x_hidden)
    else:
        x_relu = tf.nn.relu(x_hidden)

    # Apply dropout
    def f1(): return keep

    def f2(): return tf.constant(1.0)

    thresh = tf.Variable(2.0, dtype=tf.float32)
    keep_prob = tf.case([(tf.less(mode, thresh), f1)], default=f2)

    x_hidden = tf.nn.dropout(x_relu, keep_prob)

    return x_hidden


def correlation_nn(x_in, mode, keep, n_in, n_out, n1, n2):
    # Parameter decoding
    fc_1 = fc_layer(x_in, mode, n_in, n1, 1.0, 'relu')
    fc_2 = fc_layer(fc_1, mode, n1, n2, keep, 'relu')
    fc_3 = fc_layer(fc_2, mode, n2, n_out, keep, 'softmax')
    return fc_3


def mae_loss(ref, pred):
    loss = tf.losses.absolute_difference(ref, pred)
    loss = tf.reduce_mean(loss)
    return loss


def mse_loss(ref, pred):
    loss = tf.losses.mean_squared_error(ref, pred)
    loss = tf.reduce_mean(loss)
    return loss


def loss_sum(loss):
    loss_value = tf.summary.scalar('loss', loss)
    # loss_value = tf.reduce_mean(loss)
    return loss_value


def training(loss, learning_rate):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step)
    return train_op
