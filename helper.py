import numpy as np
import cv2


def imread(path):
    return cv2.imread(path)


def imrescale(image, scale):
    w, h = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def imresize(image, size):
    return cv2.resize(image, size)


def create_reals_pyramid(real, stop_scale, scale_factor):
    reals = []
    for i in range(stop_scale):
        scale = scale_factor ** (stop_scale - i)
        scaled_image = imrescale(real, scale)
        transposed_image = np.transpose(scaled_image, [2, 0, 1])
        normalized_image = np.array(transposed_image, dtype=np.float32) / 255.0
        normalized_image = (normalized_image - 0.5) * 2.0
        reals.append(np.expand_dims(normalized_image, axis=0))
    return reals
