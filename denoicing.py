import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def non_local_means_denoising(image):
    """Apply Non-Local Means Denoising."""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def bilateral_filter(image):
    """Apply Bilateral Filtering."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def pad_image(image):
    # Pad the image to ensure it is at least 7x7
    height, width = image.shape[:2]
    if height < 7 or width < 7:
        padded_image = np.pad(image, ((3, 3), (3, 3), (0, 0)), mode='constant', constant_values=0)
        return padded_image
    return image

def calculate_metrics(original, processed):
    # Ensure the images are at least 7x7 for SSIM calculation
    if original.shape[0] < 7 or original.shape[1] < 7:
        raise ValueError("Images must be at least 7x7 pixels.")

    # Calculate PSNR
    psnr_value = cv2.PSNR(original, processed)

    # Calculate SSIM, using a smaller window size if necessary
    ssim_value, _ = ssim(original, processed, full=True, multichannel=True, win_size=3)

    return psnr_value, ssim_value

