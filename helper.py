import numpy as np
import cv2
import math
import pickle


def imread(path):
    return cv2.imread(path)


def imwrite(image, path):
    cv2.imwrite(path, image)


def imrescale(image, scale):
    h, w = image.shape[:2]
    return cv2.resize(image, (math.ceil(w * scale), math.ceil(h * scale)))


def imresize(image, size):
    return cv2.resize(image, size)


def normalize(image):
    return (np.array(image, dtype=np.float32) / 255.0 - 0.5) * 2.0


def denormalize(image):
    return np.array((image / 2.0 + 0.5) * 255.0, dtype=np.uint8)


def create_reals_pyramid(real, stop_scale, scale_factor):
    reals = []
    for i in range(stop_scale + 1):
        scale = scale_factor ** (stop_scale - i)
        scaled_image = imrescale(real, scale)
        transposed_image = np.transpose(scaled_image, [2, 0, 1])
        reals.append(np.expand_dims(normalize(transposed_image), axis=0))
    return reals


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
