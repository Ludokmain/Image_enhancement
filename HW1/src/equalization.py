import cv2
import numpy as np


def equalize_lab(img_bgr):
    #Equalizzazione sul canale L di luminosità nello spazio CIE L*a*b*. 
    
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L_eq = cv2.equalizeHist(L)
    return cv2.cvtColor(cv2.merge([L_eq, a, b]), cv2.COLOR_LAB2BGR)


def equalize_hsv(img_bgr):
    #Equalizzazione sul canale V (luminosità) nello spazio HSV.
    #Preserva tinta (H) e saturazione (S).
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    V_eq = cv2.equalizeHist(V)
    return cv2.cvtColor(cv2.merge([H, S, V_eq]), cv2.COLOR_HSV2BGR)
