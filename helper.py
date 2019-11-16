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


def rescale_generated_images(images, scale):
    rets = []
    for image in images:
        denormalized_image = denormalize(image)
        transposed_image = np.transpose(denormalized_image, [1, 2, 0])
        scaled_image = imrescale(transposed_image, scale)
        normalized_image = normalize(np.transpose(scaled_image, [2, 0, 1]))
        rets.append(normalized_image)
    return np.array(rets)

def imresize(image, size):
    return cv2.resize(image, size)


def normalize(image):
    return (np.array(image, dtype=np.float32) / 255.0 - 0.5) * 2.0


def denormalize(image):
    clipped_image = np.clip(image, -1.0, 1.0)
    return np.array((clipped_image / 2.0 + 0.5) * 255.0, dtype=np.uint8)


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
