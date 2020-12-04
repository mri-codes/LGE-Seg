import numpy as np
import scipy.ndimage
import scipy.misc
import os
import matplotlib.pyplot as plt

def augment_data(images, labels):
    aug_rot=   .5
    aug_mirr=  .5
    aug_trans= .5

    images, labels = produceRandomlyMirroredImage(images, labels, aug_mirr)
    images, labels = produceRandomlyTranslatedImage(images, labels, aug_trans)
    images, labels = produceRandomlyRotatedImage(images, labels, aug_rot)

    return images, labels

def produceRandomlyTranslatedImage(images, labels, aug_trans):

    for i in range(len(images)):
        if (np.random.rand(1)[0] < aug_trans):
            translX = np.random.randint(-5, 5) # all training before Feb 13,2019 was at -10,10
            translY = np.random.randint(-5, 5)# all training before Feb 13,2019 was at -10,10
            for ii in range(np.size(images,-1)): # two or three channels
                images[i,:,:,ii] = scipy.ndimage.shift(images[i,:,:,ii],(translX, translY), mode='reflect')
            for ii in range(np.size(labels,-1)): # 3 or 4 tissues
                labels[i,:,:,ii] = scipy.ndimage.shift(labels[i,:,:,ii],(translX, translY), mode='reflect', order=0)

    return images, labels

def produceRandomlyMirroredImage(images, labels, aug_mirr):
    for i in range(len(images)):
        if (np.random.rand(1)[0] < aug_mirr):
            UpDown = np.random.randint(0, 1)
            if UpDown is 1:
                images[i,:,:,:]= np.flipud(images[i,:,:,:])
                #for j in range(num_classes):
                labels[i, :, :, :] = np.flipud(labels[i,:,:,:])
            else:
                images[i,:,:,:]= np.fliplr(images[i,:,:,:])
                #for j in range(num_classes):
                labels[i, :, :, :] = np.fliplr(labels[i,:,:,:])

    return images, labels


def produceRandomlyRotatedImage(images, labels, aug_rot):
    for i in range(len(images)):
        if (np.random.rand(1)[0] < aug_rot):
            # Get random angle between -15 and 15 degrees
            random_rotation_angle = (np.random.rand(1)[0] * 30) - 15
            images[i, :, :, :] = scipy.ndimage.rotate(images[i,:,:,:], random_rotation_angle, reshape=False)
            labels[i, :, :, :] = scipy.ndimage.rotate(labels[i, :, :, :], random_rotation_angle, reshape=False,order=0)

    return images, labels
