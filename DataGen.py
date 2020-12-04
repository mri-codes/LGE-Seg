import numpy as np
import scipy.io as sio
import scipy.misc as smsc
import scipy.ndimage
import os
import keras
from augment_3Dimages import augment_data
import hyperparams as hp
hparams = hp.create_hparams()

class ClassDataGenerator(keras.utils.Sequence):
    def __init__(self, fn_list, data_dir, batch_size=4, dim=(160,160), n_channels=2,
                 n_classes=4, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data_dir   = data_dir
        self.fn_list    = fn_list
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.fn_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        fn_list_temp = [self.fn_list[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(fn_list_temp, self.data_dir)
        X, Y = augment_data(X,Y)
        
        return X, Y
        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.fn_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, fn_list_temp, data_dir):
        # Initialization
        X = []
        Y = []

        SkipSliceFlag = []
        for i, ID in enumerate(fn_list_temp):
            im_fn= ID
            lb_fn= ID[:-4]+'_label.mat'

            if ID[3] is '_':
                lge_seq_type = 1  # magnitude LGE
            else:
                lge_seq_type = 2  # phase LGE

            dd = sio.loadmat(os.path.join(data_dir,lb_fn), mat_dtype=True)
            lbvol = dd['tmp_vol_im']
            tmp = np.zeros((self.dim[0],self.dim[0],lbvol.shape[-2],lbvol.shape[-1]))
            if lbvol.shape[0] is not self.dim[0]:
                for s in range(lbvol.shape[-1]):
                    for c in range(lbvol.shape[-2]):
                        tmp[:,:,c,s]=smsc.imresize(lbvol[:,:,c,s],self.dim, interp='nearest')
                lbvol = 3.0 * tmp / 255.0
            if self.n_classes is 3:
                lbvol[lbvol==3] = 2#replace all scar label -> myocardium

            dd = sio.loadmat(os.path.join(data_dir, im_fn), mat_dtype=True)
            imvol = dd['tmp_vol_im']
            if self.n_channels > 1:
                imvol = imvol[:, :, 0:self.n_channels, :]
            else:
                imvol = imvol[:, :, 1, :]  # only 1 channel = LGE
                imvol = np.array(imvol)[:, :, np.newaxis, :]

            tmp = np.zeros((self.dim[0],self.dim[0], self.n_channels, imvol.shape[-1]))
            if imvol.shape[0] is not self.dim[0]:
                for s in range(imvol.shape[-1]):
                    for c in range(self.n_channels):
                        tmp[:, :, c, s] = smsc.imresize(imvol[:, :, c, s], self.dim)
                imvol = tmp

            for idx in range(lbvol.shape[-1]):#loop on slices....
                # Extract one slice-image
                tmp_im = imvol[:, :, :, idx]
                tmp_im = tmp_im.astype(np.float32)
                for c in range(tmp_im.shape[-1]):
                    tmp_im[:, :, c] = (tmp_im[:, :, c] - np.min(tmp_im[:, :, c])) / (
                            (np.max(tmp_im[:, :, c]) - np.min(tmp_im[:, :, c])) + 0.001)
                X.append(tmp_im)

                # Extract one slice-GT
                tmp = np.squeeze(lbvol[:, :, 0, idx])  # 2D slice; 3rd dim is dummy
                if np.any(np.isnan(tmp)) or np.any(tmp < 0):
                    tmp = np.zeros(tmp.shape, dtype=tmp.dtype)
                    SkipSliceFlag[-1] = 1
                tmp_split = keras.utils.to_categorical(tmp, num_classes=hparams.num_tissues)  # num tissues = 4
                Y.append(tmp_split)

        XX = X
        YY = Y
        rnd_idx = np.arange(len(XX))
        np.random.shuffle(rnd_idx)
        XX = [XX[i] for i in rnd_idx]
        YY = [YY[i] for i in rnd_idx]

        # N:  batch size as # images
        N = self.batch_size * 17  # to maintain fixed number of images per batch
        if len(XX) < N:  # if up to 50% bad cases,avoid reducing the batch size severely
            XX.extend(XX)
            YY.extend(YY)
        if len(XX) < N:  # again, just in case
            XX.extend(XX)
            YY.extend(YY)

        return np.asarray(XX[0:N]), np.asarray(YY[0:N])
