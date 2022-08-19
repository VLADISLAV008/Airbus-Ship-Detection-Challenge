import numpy as np
import cv2


def rle_to_mask(rle: str, shape=(768, 768)):
    """
    :param rle: run length encoded pixels as string formated
    :param shape: (height,width) of array to return
    :return: numpy 2D array, 1 - mask, 0 - background
    """
    encoded_pixels = np.array(rle.split(), dtype=int)
    starts = encoded_pixels[::2] - 1
    ends = starts + encoded_pixels[1::2]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def mask_to_rle(img, shape=(768, 768)) -> str:
    """
    :param img: numpy 2D array, 1 - mask, 0 - background
    :param shape: (height,width) dimensions of the image
    :return: run length encoded pixels as string formated
    """
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
