import tensorflow as tf
import numpy as np
import os
import scipy.io
import corrNN_functions as TrainFuncs
import matplotlib.pyplot as plt


def main(par, ckpt_dir, test_file):
    # load, normalize and reshape test data
    test_dataset = scipy.io.loadmat(ckpt_dir + test_file)
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
        pred_op = TrainFuncs.correlation_nn(pl_sig_noisy, mode_pl, keep_pl, par['P'], par['Q'], par['nL1'], par['nL2'], par['hidden_act_fun'])

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

    mae = np.nan
    rmse = np.nan
    if "ref" in test_dataset:
        # synthetic with ground truth
        print('tested on synthetic data w/ ground truth')
        ref = test_dataset['ref']

        mae = np.mean(np.abs(ref-pred))
        print('MAE:  ' + str(mae))
        mse = np.mean(np.square(ref-pred))
        print('MSE:  ' + str(mse))
        rmse = np.sqrt(np.mean(np.square(ref-pred)))
        print('RMSE: ' + str(rmse))

        plt.figure()
        plt.imshow(np.reshape(np.mean(np.abs(ref - pred), axis=0), [60, 60]))
        plt.colorbar()
        plt.title('MAE over T1-T2-space')
        plt.savefig(ckpt_dir + 'test_synth_mae_t1t2.png')

        plt.figure()
        plt.imshow(np.reshape(np.sqrt(np.mean(np.square(ref - pred), axis=0)), [60, 60]))
        plt.colorbar()
        plt.title('RMSE over T1-T2-space')
        plt.savefig(ckpt_dir + 'test_synth_rmse_t1t2.png')

        if "no_comp" in test_dataset:
            no_comp = test_dataset['no_comp']

            ncomp_n = []
            ncomp_mae_means = []
            ncomp_mae_stds = []
            ncomp_rmse_means = []
            ncomp_rmse_stds = []
            mae_allN = np.mean(np.abs(ref - pred), axis=1)
            rmse_allN = np.sqrt(np.mean(np.square(ref - pred), axis=1))
            for ncomp in np.unique(no_comp):
                ncomp_n.append(ncomp)
                boolean_subset = np.squeeze(no_comp == ncomp)
                ncomp_mae_means.append(np.mean(mae_allN[boolean_subset]))
                ncomp_mae_stds.append(np.std(mae_allN[boolean_subset]))
                ncomp_rmse_means.append(np.mean(rmse_allN[boolean_subset]))
                ncomp_rmse_stds.append(np.std(rmse_allN[boolean_subset]))

            # bar plot for error dependency on no_comp
            fig, ax = plt.subplots()
            width = 0.35  # the width of the bars
            ind = np.arange(len(ncomp_mae_means)) + 1  # the x locations for the groups
            rects1 = ax.bar(ind, ncomp_mae_means, width, yerr=ncomp_mae_stds, label='Test')
            # rects2 = ax.bar(ind - width, ncomp_mae_means, width, yerr=ncomp_mae_stds, label='Val')
            # rects2 = ax.bar(ind + width, ncomp_mae_means, width, yerr=ncomp_mae_stds, label='Train')
            ax.set_ylabel('MAE (a.u.)')
            ax.set_title('MAE for different number of components')
            ax.set_xticks(ind)
            ax.set_xlabel('# components')
            # ax.legend()
            plt.savefig(ckpt_dir + 'test_synth_mae_comps.png')

            fig, ax = plt.subplots()
            width = 0.35  # the width of the bars
            ind = np.arange(len(ncomp_rmse_means)) + 1  # the x locations for the groups
            rects1 = ax.bar(ind, ncomp_rmse_means, width, yerr=ncomp_rmse_stds, label='Test')
            # rects2 = ax.bar(ind - width, ncomp_rmse_means, width, yerr=ncomp_rmse_stds, label='Val')
            # rects2 = ax.bar(ind + width, ncomp_rmse_means, width, yerr=ncomp_rmse_stds, label='Train')
            ax.set_ylabel('RMSE (a.u.)')
            ax.set_title('RMSE for different number of components')
            ax.set_xticks(ind)
            ax.set_xlabel('# components')
            # ax.legend()
        plt.savefig(ckpt_dir + 'test_synth_rmse_comps.png')

    else:
        # in vivo
        print('tested on in vivo data')
        pred_mean = np.mean(pred, axis=0).reshape(1, -1)
        pred_mean_arr = pred_mean[0, :].reshape([par['nT2'], par['nT1']], order='F')

        plt.figure()
        plt.imshow(pred_mean_arr)
        plt.title('average pred')
        plt.xlabel('T1 index (a.u.)')
        plt.ylabel('T2 index (a.u.)')
        plt.savefig(ckpt_dir + 'test_invivo_av_spectra.png')

        fig, ax = plt.subplots()
        t1mat, t2mat = np.meshgrid(np.linspace(50, 3000, 60), np.linspace(5, 300, 60))
        ax.contourf(t1mat, t2mat, pred_mean_arr)
        plt.title('average pred')
        plt.xlabel('T1 (ms)')
        plt.ylabel('T2 (ms)')
        ax.set_box_aspect(1)
        plt.savefig(ckpt_dir + 'test_invivo_av_spectra_contourf.png')

    return mae, rmse


if __name__ == "__main__":
    default_par = {'P': 238,
                   'Q': '',
                   'nT1': 60,
                   'nT2': 60,
                   'N': '',
                   'nL1': 1024,
                   'nL2': 2048,
                   'hidden_act_fun': 'relu'
                   }
    default_par['Q'] = default_par['nT1']*default_par['nT2']
    default_ckpt_dir = 'checkpoints/timestamp_corrNN_P=' + str(default_par['P']) + '_Q=' + str(default_par['Q']) + \
                       '_mae_ep20000_LR0.001_KR1.0_nl' + str(default_par['nL1']) + ',' + str(default_par['nL2']) + '_' + \
                       default_par['hidden_act_fun'] + '_Noise5_seed0/'
    default_ckpt_dir = 'checkpoints/202408061125_corrNN_P=238_Q=3600_mae_ep2_LR0.001_KR1.0_nl1024,2048_relu_Noise5_seed0/'
    default_test_file = 'dataset_test.mat'  # change to use in vivo data
    default_test_file = 'data_invivo.mat'
    main(default_par, default_ckpt_dir, default_test_file)

