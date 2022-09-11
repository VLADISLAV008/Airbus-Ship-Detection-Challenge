import keras.backend as K


def dice_loss(targets, inputs, smooth=1e-6):
    # flatten label and prediction.py tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


# Intersection over Union for Objects
def IoU(y_true, y_pred, tresh=1e-10):
    Intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    Union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - Intersection
    return K.mean((Intersection + tresh) / (Union + tresh), axis=0)


def IoU_loss(in_gt, in_pred):
    return 1 - IoU(in_gt, in_pred)
