import numpy as np
import cv2
from skimage.metrics import structural_similarity

def psnr(ref, img, max_val=255.0):
    # converto in float per evitare math problems (overflow)
    mse = np.mean((ref.astype(np.float64) - img.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val**2 / mse)


def ssim(ref, img):
    # SSIM tra riferimento e risultato (immagini a colori uint8).
    # channel_axis=2    l'asse 2 contiene i canali B, G, R: la SSIM viene
    #                   calcolata su ogni canale e poi si fa la media
    return structural_similarity(ref, img, channel_axis=2, data_range=255)
