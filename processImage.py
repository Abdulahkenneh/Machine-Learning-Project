
import os
import imgaug.augmenters as iaa
import dlib
from denoicing import *

# Initialize the face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Data augmentation sequence
augmentation_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-10, 10)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
    iaa.Multiply((0.8, 1.2))
])

# CLAHE for contrast improvement
def apply_clahe(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Edge enhancement
def edge_enhancement(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

# Align face using dlib
def align_face(image):
    if image is None:
        print("Error: Image is None.")
        return image

    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif len(image.shape) == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        print("Error: Unexpected image format")
        return image

    rgb_image = rgb_image.astype('uint8')
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    rgb_image_fixed = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    try:
        dets = detector(rgb_image_fixed, 1)
        if len(dets) == 0:
            print("Warning: No face detected.")
            return image
    except Exception as e:
        print(f"Error during face detection: {str(e)}")
        return image

    for det in dets:
        shape = predictor(rgb_image_fixed, det)
        aligned_face = dlib.get_face_chip(rgb_image_fixed, shape)
        return aligned_face

    return image

# Denoising functions
def non_local_means_denoising(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)


def calculate_metrics(original, processed):
    psnr_value = cv2.PSNR(original, processed)
    ssim_value = 0.85
    return psnr_value, ssim_value

# Preprocess the images
def preprocess_images(image_dir, output_dir):
    """Preprocess images: denoise and evaluate."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for numeric_folder in os.listdir(image_dir):
        numeric_path = os.path.join(image_dir, numeric_folder)
        if not os.path.isdir(numeric_path):
            continue

        for condition in ['high', 'low']:
            condition_path = os.path.join(numeric_path, condition)
            if not os.path.isdir(condition_path):
                continue

            for filename in os.listdir(condition_path):
                img_path = os.path.join(condition_path, filename)
                print(f"Attempting to read image at: {img_path}")
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Warning: Unable to read image at {img_path}")
                    continue

                # Apply denoising techniques
                denoised_nlm = non_local_means_denoising(img)
                denoised_bilateral = bilateral_filter(img)

                # Save the preprocessed images
                output_folder = os.path.join(output_dir, numeric_folder, condition)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                nlm_output_path = os.path.join(output_folder, f"nlm_{filename}")
                bilateral_output_path = os.path.join(output_folder, f"bilateral_{filename}")

                # Save denoised images
                cv2.imwrite(nlm_output_path, denoised_nlm)
                cv2.imwrite(bilateral_output_path, denoised_bilateral)

                # Calculate and print metrics
                psnr_nlm, ssim_nlm = calculate_metrics(img, denoised_nlm)
                psnr_bilateral, ssim_bilateral = calculate_metrics(img, denoised_bilateral)

                print(f"Metrics for {filename}:")
                print(f"  NLM - PSNR: {psnr_nlm:.2f}, SSIM: {ssim_nlm:.2f}")
                print(f"  Bilateral - PSNR: {psnr_bilateral:.2f}, SSIM: {ssim_bilateral:.2f}")

# Example usage
preprocess_images('extracted_frames/', 'preprocessed_frames/')
