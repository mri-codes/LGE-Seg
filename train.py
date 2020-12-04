import os
import hyperparams as hp
from DataGen import ClassDataGenerator
from DataIO import load_validation_data
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# create hyparams object
hparams = hp.create_hparams()

if hparams.volume_depth is 2:
    print('LGE-CINE FUSION is ON ')
    from models import unet_2d_shalow as umodel
else:
    print('LGE-CINE Fusion is OFF-- Optimized Baseline Model')
    from models import unet_2d_shalow_baseline as umodel

# Set data input path # Get list of image files
train_data_dir = hparams.input_dir + 'train'
label_identifier = 'label'
image_fnlist = [f for f in sorted(os.listdir(train_data_dir)) if
              os.path.isfile(os.path.join(train_data_dir, f)) and label_identifier not in f]
training_generator = ClassDataGenerator(image_fnlist, train_data_dir, n_channels=hparams.volume_depth,
                                        batch_size= hparams.batch_size, n_classes=hparams.num_tissues)

valid_data_dir = hparams.input_dir + 'valid'

vimage_fnlist = [f for f in sorted(os.listdir(valid_data_dir)) if
              os.path.isfile(os.path.join(valid_data_dir, f)) and label_identifier not in f]
valX, valY = load_validation_data(vimage_fnlist, valid_data_dir,
                                  n_classes=hparams.num_tissues, n_channels=hparams.volume_depth)

weights_dir = hparams.weights_dir

model = umodel(input_shape=(hparams.volume_width, hparams.volume_height, hparams.volume_depth),
               dropout= 0.25, num_tissues=hparams.num_tissues, learn_rate=0.005)# 0.001

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=30, min_lr=0.0005, verbose=1),
    ModelCheckpoint(weights_dir +"/model_lge_weights-R1-{epoch:02d}.hdf5", verbose=1, mode='max',
                    save_best_only=True, save_weights_only=True, monitor='val_dice_coef_myo')]

model.fit_generator(generator=training_generator, epochs=hparams.num_epochs,
                    steps_per_epoch=(len(image_fnlist) // hparams.batch_size),
                    validation_data= (valX, valY),
                    use_multiprocessing=True, callbacks=callbacks
                    , workers=2 )
