import tensorflow as tf
import numpy as np
import os
import scipy.io
import corrNN_functions as TrainFuncs


def main(par, ckpt_dir, test_file):
    # load, normalize and reshape test data
    test_dataset = scipy.io.loadmat(test_file)
    test_dataset = TrainFuncs.normalize_data(test_dataset)
    signal_noisy = test_dataset['input_noisy']
    np.transpose(signal_noisy)

    par['P'] = np.shape(signal_noisy)[1]
    par['N'] = np.shape(signal_noisy)[0]

    with tf.Graph().as_default():
        # Placeholders: input data, mode (train:1.0 or test:2.0), keep_prob
        pl_sig_noisy = tf.placeholder(tf.float32, shape=(None, par['P']))
        mode_pl = tf.placeholder(tf.float32)
        keep_pl = tf.placeholder(tf.float32)

        # Operations
        pred_op = TrainFuncs.correlation_nn(pl_sig_noisy, mode_pl, keep_pl, par['P'], par['Q'],
                                            par['NL1'], par['NL2'])

        # Create saver for saving variables
        saver = tf.train.Saver()

        # Look for the ckpt you want to use
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('ckpt.meta')]
        ckpt_file = ckpt_dir + ckpt_files[-1][:-5]
        print(ckpt_file)

        sess = tf.InteractiveSession()

        # Load the checkpoint
        saver.restore(sess, ckpt_file)

        pred = np.zeros((par['N'], par['Q']), dtype=np.float32)

        for idx in range(par['N']):
            # mode=2 because testing, keep=keep_rate
            input_dict = {pl_sig_noisy: signal_noisy[idx, :][np.newaxis, :], mode_pl: 2.0, keep_pl: 1.}

            # Run operations
            pred_out = sess.run([pred_op], feed_dict=input_dict)

            pred[idx, :] = np.asarray(pred_out)

        output_dict = {'pred': pred}
        scipy.io.savemat(ckpt_dir + test_file[:-4] + '_output.mat', output_dict)

        sess.close()


if __name__ == "__main__":
    default_par = {'P': 238,
                   'Q': '',
                   'NT1': 60,
                   'NT2': 60,
                   'N': '',
                   'NL1': 1500,
                   'NL2': 1800
                   }
    default_par['Q'] = default_par['NT1']*default_par['NT2']
    default_ckpt_dir = 'checkpoints/timestamp_corrNN_P=' + str(default_par['P']) + '_Q=' + str(default_par['Q']) + \
                       '_mae_ep20000_LR0.001_KR1.0_nl' + str(default_par['NL1']) + ',' + str(default_par['NL2']) + '_Noise5/'
    default_test_file = 'test_data_P' + str(default_par['P']) + '.mat'  # change to use in vivo data
    main(default_par, default_ckpt_dir, default_test_file)

