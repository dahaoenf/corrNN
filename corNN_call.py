import shutil
import numpy as np
from datetime import datetime
import corrNN_Training as Training
import corrNN_Testing as Testing
import corNN_GMM as GMM

timestamp = datetime.now().strftime('%Y%m%d%H%M')

T2 = np.linspace(5., 300., 60)
T1 = T2*10.

# Prepare inputs
par = {'live_testing': False,
       'noise': 5,
       'random_noise': True,
       'epochs_max': 30000,
       'stop_after_ep': 200,
       'batch_size': 4096,
       'learning_rate': 1e-3,
       'keep_rate': 1.,
       'loss': 'mae',
       'nL1': 1024,
       'nL2': 2048,
       'hidden_act_fun': 'relu',
       'seed': 0,
       'P': 238,
       'Q': '', 'nT1': '', 'nT2': '', 'nTI': '', 'nTE': '',
       }

par['nL2'] = 2*par['nL1']

invivo_file = 'test_data_invivo_P' + str(par['P']) + '.mat'
training_data_file = 'training_data_P' + str(par['P']) + '.mat'

# Perform training
par, ckpt_dir = Training.main(par, training_data_file)

# Test on synthetic data
Testing.main(par, ckpt_dir, 'dataset_test.mat')

# Apply on in vivo data
shutil.copy(invivo_file, ckpt_dir + 'data_invivo.mat')
Testing.main(par, ckpt_dir, 'data_invivo.mat')

# Evaluate in vivo results with GMM
GMM.main(ckpt_dir + 'data_invivo_output.mat', 'mask.mat', T1, T2, n_comp=8, f_threshold=1e-2)

