import numpy as np
import cv2


CONTRAST_ALPHA = 0.5   # contrast factor: 1.0 = unchanged, <1 = lower contrast
NOISE_SIGMA    = 8.0   # std. dev. of Gaussian noise, on the 0–255 scale
SEED           = 42    # fixed seed so the noise is the same every run



def degrade(img_bgr, alpha=CONTRAST_ALPHA, sigma=NOISE_SIGMA, seed=SEED):
    
    # 1) Convert to float so no math problems
    img = img_bgr.astype(np.float32)

    # 2) Contrast reduction: squeeze intensities toward the mean
    #    out = alpha * (I - mu) + mu
    mu = img.mean()                      # global mean intensity
    low_contrast = alpha * (img - mu) + mu

    # 3)  Gaussian noise: 
    rng = np.random.default_rng(seed)    # reproducible random generator 
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32) #add gaussian noise
    noisy = low_contrast + noise
    
    # 4) Clip back to the valid range and convert back to uint8
    return np.clip(noisy, 0, 255).astype(np.uint8) #return the degraded image
