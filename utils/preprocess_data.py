import os
from glob import glob

import cv2
import numpy as np
from skimage import exposure
from skimage.morphology import (
    black_tophat,
    disk,
    white_tophat
)

# ---- config

# Anisotropic diffusion 
DIFFUSION_ITERATIONS = 5
DIFFUSION_KAPPA = 20
DIFFUSION_GAMMA = 0.15

# CLAHE 
CLAHE_CLIP_LIMIT = 5.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Morphological feature extraction
TOPHAT_RADIUS = 10

# Percentile normalization range
LOW_PERCENTILE = 2
HIGH_PERCENTILE = 98


def anisotropic_diffusion(
    img, # input (grayscale) image (np.ndarray)
    num_iter=DIFFUSION_ITERATIONS, # iterations (int)
    kappa=DIFFUSION_KAPPA,  # edge sensitivity coefficient (float)
    gamma=DIFFUSION_GAMMA   # diffusion rate (float)
):

    diffused = img.astype(np.float32)

    # Iterative diffusion process
    for _ in range(num_iter):

        north = np.zeros_like(diffused)
        south = np.zeros_like(diffused)
        east = np.zeros_like(diffused)
        west = np.zeros_like(diffused)

        north[:-1, :] = diffused[1:, :] - diffused[:-1, :]
        south[1:, :] = diffused[:-1, :] - diffused[1:, :]
        east[:, :-1] = diffused[:, 1:] - diffused[:, :-1]
        west[:, 1:] = diffused[:, :-1] - diffused[:, 1:]

        c_north = np.exp(-(north / kappa) ** 2)
        c_south = np.exp(-(south / kappa) ** 2)
        c_east = np.exp(-(east / kappa) ** 2)
        c_west = np.exp(-(west / kappa) ** 2)

        # Diffusion update step
        diffused += gamma * (
            c_north * north +
            c_south * south +
            c_east * east +
            c_west * west
        )

    return diffused


def preprocess(img):
    """
    it returns -> 3-channel img [CLAHE-enhanced img; black top-hat; white top-hat] 

    """

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge-preserving denoising
    denoised = anisotropic_diffusion(gray)

    # contrast stretching using percentiles
    p_low, p_high = np.percentile(
        denoised,
        (LOW_PERCENTILE, HIGH_PERCENTILE)
    )

    contrast_enhanced = exposure.rescale_intensity(
        denoised,
        in_range=(p_low, p_high)
    )

    # normalizatoin
    normalized = cv2.normalize(
        contrast_enhanced,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # CLAHE to enhance local contrast
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    enhanced = clahe.apply(normalized)

    # Morphological operations
    structuring_element = disk(TOPHAT_RADIUS)

    black_tophat_features = black_tophat(
        enhanced,
        footprint=structuring_element
    )

    white_tophat_features = white_tophat(
        enhanced,
        footprint=structuring_element
    )

    # aaand together
    processed_img = np.stack(
        [
            enhanced,
            black_tophat_features,
            white_tophat_features
        ],
        axis=-1
    ).astype(np.uint8)

    return processed_img


def process_dataset(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob(os.path.join(input_dir, "*.*"))

    print(f"Found {len(image_paths)} images.")

    for image_path in image_paths:

        img = cv2.imread(image_path)

        if img is None:
            print(f"Skipping unreadable file: {image_path}")
            continue

        # apply preprocessing pipeline
        processed_img = preprocess(img)

        filename = os.path.basename(image_path)

        save_path = os.path.join(output_dir, filename)

        cv2.imwrite(save_path, processed_img)

        print(f"Saved: {save_path}")


if __name__ == "__main__":

    INPUT_DIR = (
        r"C:\Users\asage\Documents\.ABM"
        r"\arcade_challenge_datasets"
        r"\arcade_challenge_datasets"
        r"\dataset_phase_1"
        r"\segmentation_dataset"
        r"\seg_test"
        r"\images_noclahe"
    )

    OUTPUT_DIR = (
        r"C:\Users\asage\Documents\.ABM"
        r"\nnUNet"
        r"\nnUNet_raw"
        r"\Dataset501_ARCADE"
        r"\imagesTs"
    )

    process_dataset(INPUT_DIR, OUTPUT_DIR)