import cv2
import numpy as np

def fft_fake_score(image_bgr):
    """
    Returns a fake probability based on frequency artifacts.
    Diffusion images tend to suppress high-frequency energy.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    # FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    # Low vs High frequency regions
    low_freq = magnitude[
        center_h - h//10 : center_h + h//10,
        center_w - w//10 : center_w + w//10
    ]

    high_freq_energy = np.sum(magnitude) - np.sum(low_freq)
    total_energy = np.sum(magnitude) + 1e-8

    hf_ratio = high_freq_energy / total_energy

    # Diffusion images → abnormally LOW hf_ratio
    # Convert to fake probability
    fake_score = np.clip((0.12 - hf_ratio) * 8.0, 0.0, 1.0)

    return float(fake_score)
