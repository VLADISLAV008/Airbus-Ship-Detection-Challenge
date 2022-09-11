import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from constants import IMG_SHAPE, BUFFER_SIZE, BATCH_SIZE, IMAGES_WITHOUT_SHIPS_NUMBER, VALIDATION_LENGTH
from preprocessing.utils import rle_to_mask


def get_train_dataset() -> tuple:
    df = pd.read_csv(os.environ.get("TARGET_LABELS_DATASET"))
    df['EncodedPixels'] = df['EncodedPixels'].astype('string')

    # Delete corrupted images
    corrupted_images = ['6384c3e78.jpg']
    df = df.drop(df[df['ImageId'].isin(corrupted_images)].index)

    # Dataframe that contains the segmentation for each ship in the image.
    instances_segmentations = df

    # Dataframe that contains the segmentation of all ships in the image.
    images_segmentations = df.groupby(by=['ImageId'])['EncodedPixels'].apply(
        lambda x: np.nan if pd.isna(x).any() else ' '.join(x)).reset_index()

    return instances_segmentations, images_segmentations


def load_train_image(tensor, images_segmentations) -> tuple:
    path = tf.get_static_value(tensor).decode("utf-8")

    image_id = path.split('/')[-1]
    input_image = cv2.imread(path)
    input_image = tf.image.resize(input_image, IMG_SHAPE)
    input_image = tf.cast(input_image, tf.float32) / 255.0

    encoded_mask = images_segmentations[images_segmentations['ImageId'] == image_id].iloc[0]['EncodedPixels']
    input_mask = np.zeros(IMG_SHAPE + (1,))
    if not pd.isna(encoded_mask):
        input_mask = rle_to_mask(encoded_mask)
        input_mask = cv2.resize(input_mask, IMG_SHAPE, interpolation=cv2.INTER_AREA)
        input_mask = np.expand_dims(input_mask, axis=2)

    return input_image, input_mask


def prepare_train_and_validation_batches():
    _, images_segmentations = get_train_dataset()

    # reduce the number of images without ships
    images_without_ships = images_segmentations[images_segmentations['EncodedPixels'].isna()]['ImageId'].values
    images_without_ships = images_without_ships[:IMAGES_WITHOUT_SHIPS_NUMBER]
    images_with_ships = images_segmentations[images_segmentations['EncodedPixels'].notna()]['ImageId'].values
    images_list = np.append(images_without_ships, images_with_ships)

    images_list = tf.data.Dataset.list_files([f'{os.environ.get("TRAIN_DIR")}{name}' for name in images_list])

    train_images = images_list.map(
        lambda x: tf.py_function(load_train_image, [x, images_segmentations], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE)

    validation_dataset = train_images.take(VALIDATION_LENGTH)
    train_dataset = train_images.skip(VALIDATION_LENGTH)

    train_batches = (
        train_dataset
            .shuffle(BUFFER_SIZE)
            .repeat()
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

    validation_batches = (
        validation_dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))
    return train_batches, validation_batches
