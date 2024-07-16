import numpy as np
from scipy.stats import skew, kurtosis

def statistical_analysis(img):
    mean_value = np.mean(img)
    std_dev_value = np.std(img)
    skewness_value = skew(img.flatten())
    kurtosis_value = kurtosis(img.flatten())
    
    return mean_value, std_dev_value, skewness_value, kurtosis_value