import cv2
import numpy as np

from constants import IMG_SHAPE
from models.u_net_model import create_model


def load_model(weights_path: str):
    model = create_model()
    model.load_weights(weights_path)
    return model


def preprocessing_image(image):
    image = cv2.resize(image, IMG_SHAPE, interpolation=cv2.INTER_AREA)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict(model, image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError('Image not found error')

    image = preprocessing_image(image)
    pred_mask = model.predict(image)[0]
    cv2.imshow('image', pred_mask)
    cv2.waitKey(0)


model = load_model('./../checkpoints/model-checkpoint')
predict(model, './../images/0010551d9.jpg')
