import numpy as np
import scipy.io as sio
import scipy.misc as smsc
import scipy.ndimage
import os
import keras
import krs_hyperparams as hp
hparams = hp.create_hparams()

def load_validation_data(fn_list, data_dir, im_size=(160,160),n_channels=2,n_classes=4):
    # Initialization
    X = []
    Y = []
    glob_idx = -1
    # Generate data ## LOAD MAT FILES.....
    for i, ID in enumerate(fn_list):
        im_fn = ID
        lb_fn = ID[:-4] + '_label.mat'
        ddd = sio.loadmat(os.path.join(data_dir, lb_fn), mat_dtype=True)
        lbvol = ddd['tmp_vol_im']
        tmp = np.zeros((im_size[0], im_size[0], lbvol.shape[-2], lbvol.shape[-1]))
        if lbvol.shape[0] is not im_size[0]:
            for s in range(lbvol.shape[-1]):
                for c in range(lbvol.shape[-2]): #only one channel
                    tmp[:, :, c, s] = smsc.imresize(lbvol[:, :, c, s], im_size, interp='nearest')
            lbvol = 3.0*tmp/255.0
        if n_classes is 3:
            lbvol[lbvol == 3] = 2

        dd = sio.loadmat(os.path.join(data_dir, im_fn), mat_dtype=True)
        imvol = dd['tmp_vol_im']
        if n_channels > 1:
            imvol =imvol[:,:,0:n_channels,:]
        else:
            imvol = imvol[:, :, 1, :]#only 1 channel = LGE
            imvol = np.array(imvol)[:, :, np.newaxis, :]

        tmp = np.zeros((im_size[0], im_size[0], n_channels, imvol.shape[-1]))
        if imvol.shape[0] is not im_size[0]:
            for s in range(imvol.shape[-1]):
                for c in range(n_channels):
                    tmp[:, :, c, s] = smsc.imresize(imvol[:, :, c, s], im_size)
            imvol = tmp

        for idx in range(lbvol.shape[-1]):  # loop on slices....
            glob_idx = glob_idx + 1
            # Extract one slice-image; per-channel normalization
            tmp_im = imvol[:, :, :, idx]
            tmp_im = tmp_im.astype(np.float32)
            for c in range(tmp_im.shape[-1]): #normalize
                tmp_im[:, :, c] = (tmp_im[:, :, c] - np.min(tmp_im[:, :, c])) / ((np.max(tmp_im[:, :, c]) - np.min(tmp_im[:, :, c])) + 0.001)
            X.append(tmp_im)

            # Extract one slice-GT
            tmp = np.squeeze(lbvol[:, :, 0, idx])  # 2D slice; 3rd dim is dummy
            if np.any(np.isnan(tmp)) or np.any(tmp < 0):
                tmp = np.zeros(tmp.shape, dtype=tmp.dtype)
                SkipSliceFlag[-1]=1
            tmp_split = keras.utils.to_categorical(tmp, num_classes=hparams.num_tissues)  # num tissues = 4
            Y.append(tmp_split)

    XX = X
    YY = Y

    return np.asarray(XX), np.asarray(YY)

def load_patient(fn_list, data_dir, pat_index=0, im_size=(160,160),n_channels=1,n_classes=4, rot_angle=0):
    # Initialization
    X = []
    Y = []
    Sloc = []
    # Generate data ## LOAD MAT FILES.....
    ID = fn_list[pat_index]
    im_fn = ID
    lb_fn = ID[:-4] + '_label.mat'

    dd = sio.loadmat(os.path.join(data_dir, im_fn), mat_dtype=True)
    imvol = dd['tmp_vol_im']
    if n_channels > 1:
        imvol = imvol[:, :, 0:n_channels, :]
    else:
        imvol = imvol[:, :, 1, :]  # only 1 channel = LGE
        imvol = np.array(imvol)[:, :, np.newaxis, :]

    tmp = np.zeros((im_size[0], im_size[0], n_channels, imvol.shape[-1]))
    if imvol.shape[0] is not im_size[0]:
        for s in range(imvol.shape[-1]): #slices
            for c in range(n_channels):
                tmp[:, :, c, s] = smsc.imresize(imvol[:, :, c, s], im_size)
        imvol = tmp

    for idx in range(imvol.shape[-1]):  # loop on slices....
        tmp_im = imvol[:, :, :, idx]
        tmp_im = tmp_im.astype(np.float32)
        for c in range(tmp_im.shape[-1]):
            tmp_im[:, :, c] = (tmp_im[:, :, c] - np.min(tmp_im[:, :, c])) / (
                        (np.max(tmp_im[:, :, c]) - np.min(tmp_im[:, :, c])) + 0.001)
        X.append(tmp_im)

    ddd = sio.loadmat(os.path.join(data_dir, lb_fn), mat_dtype=True)
    lbvol = ddd['tmp_vol_im']
    tmp = np.zeros((im_size[0], im_size[0], lbvol.shape[-2], lbvol.shape[-1]))
    if lbvol.shape[0] is not im_size[0]:
        for s in range(lbvol.shape[-1]): # only one channel
            for c in range(lbvol.shape[-2]):
                tmp[:, :, c, s] = smsc.imresize(lbvol[:, :, c, s], im_size, interp='nearest')
        lbvol = 3.0 * tmp / 255.0
    if n_classes is 3:
        lbvol[lbvol == 3] = 2

    for idx in range(lbvol.shape[-1]):  # loop on slices....
        tmp = np.squeeze(lbvol[:, :, :, idx])  # 3D array
        if np.any(np.isnan(tmp)) or np.any(tmp < 0):
            tmp = np.zeros(tmp.shape, dtype=tmp.dtype)
        tmp_split = keras.utils.to_categorical(tmp, num_classes=hparams.num_tissues)
        Y.append(tmp_split)

    return np.asarray(X), np.asarray(Y)

