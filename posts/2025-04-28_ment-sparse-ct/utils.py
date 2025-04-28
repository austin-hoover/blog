import os
import numpy as np
import skimage


def radon_transform(image: np.ndarray, angles: np.ndarray) -> np.ndarray:
    image = np.copy(image)
    image = image.T
    theta = -np.degrees(angles)
    sinogram = skimage.transform.radon(image, theta=theta)
    return sinogram


def rec_sart(sinogram: np.ndarray, angles: np.ndarray, iterations: int = 1) -> np.ndarray:
    theta = -np.copy(np.degrees(angles))
    image = skimage.transform.iradon_sart(sinogram, theta=theta)
    for _ in range(iterations - 1):
        image = skimage.transform.iradon_sart(sinogram, theta=theta, image=image)
    image = image.T
    return image


def rec_fbp(sinogram: np.ndarray, angles: np.ndarray, iterations: int = 1) -> np.ndarray:
    theta = -np.copy(np.degrees(angles))
    image = skimage.transform.iradon(sinogram, theta=theta)
    image = image.T
    return image


def load_image(res: int = None, blur: float = 0.0, pad: int = 25) -> None:
    image = skimage.io.imread("./images/tree.png", as_gray=True)
    image = 1.0 - image
    image = image[::-1, :]
    image = image.T

    pad = max(pad, 25)

    if pad:    
        shape = image.shape
        new_shape = tuple(np.add(shape, pad * 2))
        new_image = np.zeros(new_shape)
        new_image[pad:-pad, pad:-pad] = image.copy()
        image = np.copy(new_image)
        
    if res:
        shape = (res, res)
        image = skimage.transform.resize(image, shape, anti_aliasing=True)
        
    if blur:
        image = skimage.filters.gaussian(image, blur)
        
    return image
