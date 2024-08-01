import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from sklearn.mixture import GaussianMixture
import statsmodels


def main(result_file, mask_file, T1, T2, n_comp=8, f_threshold=1e-2):
    # load data
    data = sio.loadmat(result_file)
    mask = sio.loadmat(mask_file)['mask']
    print(data.keys())  # should contain 'pred'

    # output folder for plots
    colormap_cycle = ['Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'BuGn', 'PuBu', 'PuRd', 'YlOrBr']
    plt_dir = os.path.dirname(result_file)
    if not os.path.isdir(plt_dir):
        os.mkdir(plt_dir)

    # spectra - masked and thresholded
    F = np.reshape(data['pred'], [np.shape(mask)[0], np.shape(mask)[1], np.size(T1), np.size(T2)], order='F')
    F_thresh = F.copy()
    F_thresh[F_thresh < f_threshold] = 0

    # convert spectra into counts
    C_thresh = F_thresh * 1. / F_thresh[F_thresh > 0].min()
    C_thresh = np.round(C_thresh)
    C_thresh = C_thresh.astype(int)

    C_thresh_mask = C_thresh[mask > 0]
    print(C_thresh.shape)
    print(C_thresh_mask.shape)

    C_thresh_mask_sum = np.sum(C_thresh_mask, axis=0)
    C_thresh_mask_sum[C_thresh_mask_sum < 1000] = 0

    # Put counts into T1-T2-list
    t1_thresh = []
    t2_thresh = []
    for r in range(60):
        for c in range(60):
            t1_thresh = t1_thresh + [r] * C_thresh_mask_sum[r, c]
            t2_thresh = t2_thresh + [c] * C_thresh_mask_sum[r, c]

    # Format for scatter
    X = np.zeros([len(t1_thresh), 2])
    X[:, 0] = t1_thresh
    X[:, 1] = t2_thresh

    # run GMM
    gm = GaussianMixture(n_components=n_comp, covariance_type='full', verbose=True).fit(X)
    labels = gm.predict(X)
    labels = labels + 1  # to set only background to 0

    # Go back to physical ranges
    X_phys = X.copy()
    for t1_ind, t1 in enumerate(T1):
        X_phys[X[:, 1] == t1_ind, 0] = t1
    for t2_ind, t2 in enumerate(T2):
        X_phys[X[:, 0] == t2_ind, 1] = t2

    # plot of all gaussians for average spectrum
    plt.figure(figsize=(5, 5))
    for i in range(1, n_comp + 1):
        ind = np.where(labels == i)
        ind = np.random.permutation(ind[0])
        # exemplary_voxels = 1000  # optional: plot just a part of the data to speed up plotting
        # ind = ind[:exemplary_voxels]  # optional: plot just a part of the data to speed up plotting
        df = pd.DataFrame(data={'T1 (ms)': X_phys[ind, 0], 'T2 (ms)': X_phys[ind, 1]})
        print(df.shape)
        # kdeplot only works with seaborn==0.10.1 and statsmodels installed
        sns.kdeplot(df['T1 (ms)'], df['T2 (ms)'], shade=True, shade_lowest=False, alpha=0.6, antialiased=True, bw=[np.min(T1), np.min(T2)], cmap=colormap_cycle[i-1])
        print(plt.get_cmap())
    plt.xlim(0, np.max(T1))
    plt.ylim(0, np.max(T2))
    plt.xlabel('T1 (ms)', size=14)
    plt.ylabel('T2 (ms)', size=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(result_file.replace('.mat', '_GMM.pdf'), bbox_inches='tight', format='pdf', dpi=300)

    # voxel wise labeling
    label_array = np.zeros([128, 128, n_comp])
    label_array_mask = np.zeros([C_thresh_mask.shape[0], n_comp])

    count_skip = 0

    for i in range(C_thresh_mask.shape[0]):
        # print(i)
        t1_thresh_test = []
        t2_thresh_test = []
        try:
            for r in range(60):
                for c in range(60):
                    t1_thresh_test = t1_thresh_test + [r] * C_thresh_mask[i, r, c]  # [T1[0][r]]*C_thresh_mask[i,r, c]
                    t2_thresh_test = t2_thresh_test + [c] * C_thresh_mask[i, r, c]  # [T2[0][c]]*C_thresh_mask[i,r, c]

            X_test = np.zeros([len(t1_thresh_test), 2])
            X_test[:, 0] = t1_thresh_test
            X_test[:, 1] = t2_thresh_test

            labels_out = gm.predict(X_test)
            counts_total = labels_out.shape[0]
            for n in range(n_comp):
                count = np.where(labels_out == n)[0]

                try:
                    label_array_mask[i, n] = count.shape[0] / counts_total
                except ZeroDivisionError:
                    label_array_mask[i, n] = 0.
        except:
            count_skip = count_skip + 1

    print(count_skip)
    label_array[mask > 0, :] = label_array_mask

    # plot compartmental volume fraction maps
    for n in range(n_comp):
        plt.figure(figsize=(7, 5))
        plt.imshow(label_array[:, :, n], vmin=0, vmax=1)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=20)
        plt.axis('off')
        plt.savefig(result_file.replace('.mat', '_GMM_comp' + str(n+1) + '.pdf'),
                    bbox_inches='tight', format='pdf', dpi=300)


if __name__ == "__main__":
    in_file = 'checkpoints/timestamp_corrNN_P=238_Q=3600_mae_ep20000_LR0.001_KR1.0_nl1024,2048_Noise5/test_data_P238_output.mat'
    mask_file = 'mask.mat'
    T1 = np.linspace(50., 3000., 60)
    T2 = T1/10.
    n_comp = 8
    f_threshold = 1e-2
    main(in_file, mask_file, T1, T2, n_comp, f_threshold)
