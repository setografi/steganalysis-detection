import numpy as np
import pywt

def transform_domain_analysis(img):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    energy = np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)
    
    return energy