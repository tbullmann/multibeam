import numpy as np
from numpy.fft import fft, ifft
from scipy.stats import pearsonr


def snr_from_corr(x, y):
    """
    Estimate signal to noise ratio from two different versions of the same signal contaminated by uncorrelated noise,
    using the sample cross-correlation coefficient (Pearson correlation coefficient).

        snr = np.exp(-2/(N-3))*(r_N/(1-r_N)+1/2)-1/2    (for small N, assuming normal distribution and same variance)
        snr = r_N / (1 - r_N)                           (for large N>10000)

    Note: Frank and Al-Ali, 1975, Nature
    :param x, y: 1D arrays
    :return: snr: signal to noise ratio
    """
    N = len(x)
    r_N, _ = pearsonr(x, y)
    if N > 10000:
        snr = r_N / (1 - r_N)
    else:
        snr = np.exp(-2/(N-3))*(r_N/(1-r_N)+1/2)-1/2
    return snr


def snr_from_autocorr(img, axis=1):
    """
    Estimate signal to noise ratio from the autocorrelation, assuming noise contributes only to the zero lag value:

        snr = (r[1] - mu) / (r[0] - r[1])

    Note: Kim et al, 2003, J Microscop
    :param img: ndarray
    :param axis: should be perpendicular to scanning direction to avoid noise correlation within in scan direction
    :return: snr: signal to noise ratio
    """
    imgFT = fft(img - np.mean(img), axis=axis)
    imgAC = ifft(imgFT * np.conjugate(imgFT), axis=axis).real
    r = np.median(imgAC, axis=1-axis)   # median is more robust
    mu = r[len(r) // 2]
    snr = (r[1] - mu) / (r[0] - r[1])
    return snr
