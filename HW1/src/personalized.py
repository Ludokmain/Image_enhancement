import cv2
import numpy as np


def denoise_then_stretch(img_bgr, h=3, h_color=9, p_low=1, p_high=99, chroma_gain=2):
    # Denoising prima dello stretching, per non amplificare il rumore
    den = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=h, hColor=h_color,
                                          templateWindowSize=7, searchWindowSize=21)
    # Stretching solo sulla luminanza per non alterare il colore
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    lo, hi = (float(v) for v in np.percentile(L, (p_low, p_high)))
    if hi - lo > 1e-6:                      
        L = (L - lo) * 255.0 / (hi - lo)

    # 4) Amplificazione della croma: allontano a e b dal grigio neutro (128 in OpenCV).
    a = 128.0 + chroma_gain * (a - 128.0)
    b = 128.0 + chroma_gain * (b - 128.0)

    lab = cv2.merge([L, a, b])
    lab = np.clip(np.round(lab), 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)