import tensorflow as tf
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import corrNN_functions as Funcs
import time
import random
from datetime import datetime


def main(par, input_file):
    # Set up location and parameters
    timestamp = datetime.now().strftime('%Y%m%d%H%M')

    # load data (already split into training, validation, testing)
    dataset_train, dataset_val, dataset_test, t1_t2_combs, ti_te_combs = Funcs.load_split_dataset(input_file)
    dataset_train.keys()

    par['P'] = np.shape(dataset_train['input'])[1]
    par['Q'] = np.shape(dataset_train['ref'])[1]

    par['nT1'] = len(np.unique(t1_t2_combs[:, 0]))
    par['nT2'] = len(np.unique(t1_t2_combs[:, 1]))
    par['nTI'] = len(np.unique(ti_te_combs[:, 0]))
    par['nTE'] = len(np.unique(ti_te_combs[:, 1]))

    # add noise
    random.seed(par['seed'])
    np.random.seed(par['seed'])
    dataset_train = Funcs.add_noise(dataset_train, par['noise'], par['random_noise'])
    dataset_val = Funcs.add_noise(dataset_val, par['noise'], par['random_noise'])
    dataset_test = Funcs.add_noise(dataset_test, par['noise'], par['random_noise'])

    # normalize
    dataset_train = Funcs.normalize_data(dataset_train)
    dataset_val = Funcs.normalize_data(dataset_val)
    dataset_test = Funcs.normalize_data(dataset_test)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    steps_per_epoch_train = dataset_train['input'].shape[0] // par['batch_size']
    steps_per_epoch_val = dataset_val['input'].shape[0] // par['batch_size']
    steps_per_epoch_test = dataset_test['input'].shape[0] // par['batch_size']

    print('steps per epoch and val', steps_per_epoch_train, steps_per_epoch_val)
    train_losses = np.zeros(par['epochs_max'])
    val_losses = np.zeros(par['epochs_max'])
    test_losses = np.zeros(par['epochs_max'])

    train_random_idxs = list(range(dataset_train['input'].shape[0]))
    train_random_idxs = np.random.permutation(train_random_idxs)
    val_random_idxs = list(range(dataset_val['input'].shape[0]))
    val_random_idxs = np.random.permutation(val_random_idxs)
    test_random_idxs = list(range(dataset_test['input'].shape[0]))
    test_random_idxs = np.random.permutation(test_random_idxs)

    label = timestamp + '_corrNN_P=' + str(par['P']) + '_Q=' + str(par['Q']) + '_' + par['loss'] + '_ep' + str(par['epochs_max']) + \
            '_LR' + str(par['learning_rate']) + '_KR' + str(par['keep_rate']) + '_nl' + str(par['NL1']) + ',' + str(par['NL2']) + \
            '_Noise' + str(par['noise']) + '_seed' + str(par['seed'])

    ckpt_dir = 'checkpoints/' + label + '/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # save dataset for later analysis
    # sio.savemat(ckpt_dir + 'dataset.mat', dataset)
    dataset_split = {'dataset_train': dataset_train,
                     'dataset_val': dataset_val,
                     'dataset_test': dataset_test
                     }
    sio.savemat(ckpt_dir + 'dataset_split.mat', dataset_split)

    sio.savemat(ckpt_dir + 'dataset_test.mat', dataset_test)

    # to monitor loss decrease
    ep_since_last_val_improv = 0
    start_time = time.time()
    with tf.Graph().as_default():
        tf.random.set_random_seed(par['seed'])

        # Placeholders for input data
        pl_sig_noisy = tf.placeholder(tf.float32, shape=(None, par['P']))
        pl_ref = tf.placeholder(tf.float32, shape=(None, par['Q']))

        # Placeholder for keep_prob in dropout (mode: train:1.0 or test:2.0)
        mode_pl = tf.placeholder(tf.float32)
        keep_pl = tf.placeholder(tf.float32)

        # Operations
        pred_op = Funcs.correlation_nn(pl_sig_noisy, mode_pl, keep_pl, par['P'], par['Q'], par['NL1'], par['NL2'])

        if par['loss'] == 'mse':
            loss_op = Funcs.mse_loss(pl_ref, pred_op)
        else:
            # default loss is mean absolute error
            loss_op = Funcs.mae_loss(pl_ref, pred_op)

        train_op = Funcs.training(loss_op, par['learning_rate'])
        loss_sum_op = Funcs.loss_sum(loss_op)

        # Keep model with the best validation loss
        best_val_loss = 1e10

        # Create saver for saving variables; max_to_keep -> ten best checkpoints
        saver = tf.train.Saver(max_to_keep=par['epochs_max'])

        # Initialize the variables
        init = tf.global_variables_initializer()

        sess = tf.InteractiveSession()
        sess.run(init)

        # Save results for plotting on tensorboard
        tf.summary.scalar('train_loss', loss_op)
        train_loss_writer = tf.summary.FileWriter(ckpt_dir + 'train_loss_summary', sess.graph)

        tf.summary.scalar('val_loss', loss_op)
        val_loss_writer = tf.summary.FileWriter(ckpt_dir + 'val_loss_summary', sess.graph)

        if par['live_testing']:
            tf.summary.scalar('test_loss', loss_op)
            test_loss_writer = tf.summary.FileWriter(ckpt_dir + 'test_loss_summary', sess.graph)

        for ep in range(par['epochs_max']):

            # Training
            total_train_loss = 0
            for step in range(steps_per_epoch_train):
                # Get batches of input and ref vectors
                batch_start_idx = step * par['batch_size']
                batch_ind = range(batch_start_idx, batch_start_idx + par['batch_size'])
                signal_noisy, signal_clean, ref = Funcs.get_batch(dataset_train, batch_ind, train_random_idxs)

                # Create input dictionary to feed network
                input_dict = {pl_sig_noisy: signal_noisy, pl_ref: ref, mode_pl: 1.0, keep_pl: par['keep_rate']}

                # Run operations
                activations, pred_train, train_loss, train_loss_sum = sess.run([train_op, pred_op, loss_op, loss_sum_op], feed_dict=input_dict)

                train_loss_writer.add_summary(train_loss_sum, ep * steps_per_epoch_train + step)

                total_train_loss = total_train_loss + train_loss

            total_train_loss = total_train_loss / steps_per_epoch_train
            train_losses[ep] = total_train_loss

            # Validation
            total_val_loss = 0
            for step in range(steps_per_epoch_val):
                # Get batches of input and ref vectors
                val_batch_start_idx = step * par['batch_size']
                val_batch_ind = range(val_batch_start_idx, val_batch_start_idx + par['batch_size'])
                signal_noisy_val, signal_clean_val, ref_val = Funcs.get_batch(dataset_val, val_batch_ind, val_random_idxs)

                # Create input dictionary to feed network
                val_input_dict = {pl_sig_noisy: signal_noisy_val, pl_ref: ref_val, mode_pl: 2.0, keep_pl: par['keep_rate']}

                # Run operations
                activations, pred_val, val_loss, val_loss_sum = sess.run([train_op, pred_op, loss_op, loss_sum_op], feed_dict=val_input_dict)

                val_loss_writer.add_summary(val_loss_sum, ep * steps_per_epoch_train + step)

                total_val_loss = total_val_loss + val_loss

            total_val_loss = total_val_loss / steps_per_epoch_val
            val_losses[ep] = total_val_loss

            if par['live_testing']:
                # Testing
                total_test_loss = 0
                for step in range(steps_per_epoch_test):
                    # Get batches of input and ref vectors
                    test_batch_start_idx = step * par['batch_size']
                    test_batch_ind = range(test_batch_start_idx, test_batch_start_idx + par['batch_size'])
                    signal_noisy_test, signal_clean_test, ref_test = Funcs.get_batch(dataset_test, test_batch_ind, test_random_idxs)

                    # Create input dictionary to feed network
                    test_input_dict = {pl_sig_noisy: signal_noisy_test, pl_ref: ref_test, mode_pl: 2.0, keep_pl: par['keep_rate']}

                    # Run operations
                    activations, pred_test, test_loss, test_loss_sum = sess.run([train_op, pred_op, loss_op, loss_sum_op], feed_dict=test_input_dict)

                    test_loss_writer.add_summary(test_loss_sum, ep * steps_per_epoch_test + step)

                    total_test_loss = total_test_loss + test_loss

                total_test_loss = total_test_loss / steps_per_epoch_test
                test_losses[ep] = total_test_loss

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_epoch = ep
                saver.save(sess, ckpt_dir + 'model.ckpt')
                ep_since_last_val_improv = 0

                print("Val loss improved:")
                print('Training loss: \t{:.9f}*1e-3'.format(total_train_loss * 1e3))
                print('Validation loss:{:.9f}*1e-3'.format(total_val_loss * 1e3))
                if par['live_testing']:
                    print('Testing loss: \t{:.9f}*1e-3'.format(total_test_loss * 1e3))

                if par['live_testing']:
                    plt_idx = random.randint(0, np.shape(ref_test)[0])
                    ref_test_arr = ref_test[plt_idx, :].reshape(par['nT2'], par['nT1'])
                    pred_test_arr = pred_test[plt_idx, :].reshape(par['nT2'], par['nT1'])
                    plt.subplot(1, 2, 1)
                    plt.gca().set_title('ref (test)')
                    plt.imshow(ref_test_arr)
                    plt.subplot(1, 2, 2)
                    plt.gca().set_title('pred (test)')
                    plt.imshow(pred_test_arr)
                    plt.savefig(ckpt_dir + 'epoch_' + str(ep) + '.png')
                else:
                    plt_idx = random.randint(0, np.shape(ref_val)[0])
                    ref_val_arr = ref_val[plt_idx, :].reshape(par['nT2'], par['nT1'])
                    pred_val_arr = pred_val[plt_idx, :].reshape(par['nT2'], par['nT1'])
                    plt.subplot(1, 2, 1)
                    plt.gca().set_title('ref (val)')
                    plt.imshow(ref_val_arr)
                    plt.subplot(1, 2, 2)
                    plt.gca().set_title('pred (val)')
                    plt.imshow(pred_val_arr)
                    plt.savefig(ckpt_dir + 'epoch_' + str(ep) + '.png')
            else:
                ep_since_last_val_improv = ep_since_last_val_improv + 1
                if ep_since_last_val_improv >= par['stop_after_ep']:
                    print('Stopping after ' + str(ep_since_last_val_improv) + ' epochs without improvement')
                    break

            train_random_idxs = np.random.permutation(train_random_idxs)
            val_random_idxs = np.random.permutation(val_random_idxs)
            print('{:05.0f} {:05.0f} \t{:.9f}*1e-3'.format(ep, ep_since_last_val_improv, total_val_loss * 1e3))

        elapsed_time = time.time() - start_time
        print('Elapsed: {:.0f}s'.format(elapsed_time))
        print('Model at epoch ' + str(best_epoch) + ' saved with best validation loss of {:.9f}*1e-3'.format(best_val_loss*1e3))

        train_loss_writer.close()
        val_loss_writer.close()
        if par['live_testing']:
            test_loss_writer.close()

        sess.close()
        print('CLOSED SESS')

        fig, ax = plt.subplots()
        ax.plot(range(ep), train_losses[0:ep], 'r-', label='training')
        ax.plot(range(ep), val_losses[0:ep], 'b-', label='validation')
        if par['live_testing']:
            ax.plot(range(ep), test_losses[0:ep], 'g-', label='testing')
        ax.legend()
        ax.set(xlabel='#epoch', ylabel='loss', title='training vs validation vs testing loss')
        ax.grid()
        plt.savefig(ckpt_dir + 'losses.png')

        np.save(ckpt_dir + 'train_loss.npy', train_losses)
        np.save(ckpt_dir + 'val_loss.npy', val_losses)
        if par['live_testing']:
            np.save(ckpt_dir + 'test_loss.npy', test_losses)

    return par, ckpt_dir


if __name__ == "__main__":
    default_par = {'live_testing': False,
                   'noise': 5,
                   'random_noise': True,
                   'epochs_max': 20000,
                   'stop_after_ep': 200,
                   'batch_size': 4096,
                   'learning_rate': 1e-3,
                   'keep_rate': 1.,
                   'loss': 'mae',
                   'NL1': 1024,
                   'NL2': 2048,
                   'seed': 0,
                   'P': 238,
                   'Q': '', 'nT1': '', 'nT2': '', 'nTI': '', 'nTE': '',
                   }
    default_input_file = 'training_data_P' + str(default_par['P'])
    main(default_par, default_input_file)
