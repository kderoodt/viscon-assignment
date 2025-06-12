import cv2
import os
import numpy as np
from pathlib import Path
import random
import shutil

input_folder = "images_viscon"
output_folder = "images_viscon_processed"

clip_limit = 4.0
tile_grid_size = (8, 8)
gaussian_kernel = (5, 5)

Path(output_folder).mkdir(parents=True, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Could not read image {filename}")
            continue

        h = img.shape[0]
        img_cropped = img[int(0.4 * h):, :]

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_clahe = clahe.apply(img_cropped)

        img_blur = cv2.GaussianBlur(img_clahe, gaussian_kernel, 0)
        img_norm = img_blur.astype(np.float32) / 255.0

        base_name = os.path.splitext(filename)[0]

        img_to_save = (img_norm * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, base_name + ".png"), img_to_save)

print("All images processed and saved.")

source_folder = "images_viscon_processed"
destination_folder = os.path.join(source_folder, "dl")
sample_size = 200

Path(destination_folder).mkdir(parents=True, exist_ok=True)

image_files = [f for f in os.listdir(source_folder)
               if f.lower().endswith((".png", ".jpg", ".jpeg", ".npy")) and
               os.path.isfile(os.path.join(source_folder, f))]

if len(image_files) < sample_size:
    raise ValueError(f"Not enough images in folder: found {len(image_files)}, need {sample_size}")

selected_files = random.sample(image_files, sample_size)

for file_name in selected_files:
    src_path = os.path.join(source_folder, file_name)
    dst_path = os.path.join(destination_folder, file_name)
    shutil.copy2(src_path, dst_path)

print(f"Copied {sample_size} images to '{destination_folder}'.")
