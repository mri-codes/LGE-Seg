import os
import hyperparams as hp
from DataIO import load_patient
from keras.utils import plot_model
from models import dice_coef_eval
import numpy as np
import scipy.io as sio
import scipy.ndimage

import time

# create hyparams object
hparams = hp.create_hparams()

if hparams.volume_depth is 2:
    print('LGE-CINE FUSION is ON ')
    from krs_model2d import unet_2d_shalow as umodel
else:
    print('LGE-CINE Fusion is OFF-- Optimized Baseline Model')
    from krs_model2d import unet_2d_shalow_baseline as umodel

# Set data input path # Get list of image files
label_identifier = 'label'
test_data_dir = hparams.input_dir + 'test'

test_image_fnlist = [f for f in sorted(os.listdir(test_data_dir)) if
              os.path.isfile(os.path.join(test_data_dir, f)) and label_identifier not in f]
# Set path to save images to during training
results_dir_base = os.path.join(hparams.results_dir, 'testing')
weights_dir = hparams.weights_dir
model = umodel(input_shape=(hparams.volume_width, hparams.volume_height, hparams.volume_depth),
               dropout= 0.25, num_tissues=hparams.num_tissues)
model.load_weights(weights_dir + hparams.weights_for_testing)

vol_in = []
vol_gt = []
vol_prd=[]

pat =0
while pat < len(test_image_fnlist):
    start_time = time.time()
    testX, testY = load_patient(test_image_fnlist, test_data_dir, pat_index=pat,
                                n_channels=hparams.volume_depth, n_classes=hparams.num_tissues)
    vol_gt = testY
    vol_in = testX
    vol_prd = model.predict(rot_testX)

    pat_fn = test_image_fnlist[pat]
    if hparams.attn_type is 'NONE':
        sio.savemat(results_dir_base + '/P-'+str(pat)+'_' +pat_fn + '_img_pred_gt.mat',
                {"vol_in": vol_in, "vol_gt": vol_gt, "vol_prd": vol_prd})

    print("--- %s seconds ---" % ((time.time() - start_time)/np.size(vol_in,axis=0)))
    print("--- %s num slices ---" % (np.size(vol_in, axis=0)))
    pat = pat + 1
