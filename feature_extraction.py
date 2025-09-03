import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Path to dataset
dataset_path = r"C:\Users\Yoked\Desktop\Rice project\Rice_Image_Dataset"
output_csv = "rice_features.csv"

all_features = []

# Count total images for progress bar
total_images = sum([len([f for f in os.listdir(os.path.join(dataset_path, label))
                         if f.lower().endswith(('.jpg', '.png'))])
                    for label in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, label))])

pbar = tqdm(total=total_images, desc="Extracting features")

# Loop over each class folder
for label in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, label)
    if not os.path.isdir(class_path):
        continue

    # Loop over each image in the class folder
    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)
        if not (file.lower().endswith(".jpg") or file.lower().endswith(".png")):
            continue

        img = cv2.imread(img_path)
        if img is None:
            pbar.update(1)
            continue

        # Convert to grayscale and threshold to binary mask
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find external contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            pbar.update(1)
            continue
        cnt = max(contours, key=cv2.contourArea)

        # Shape features
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Fit ellipse if possible
        if len(cnt) >= 5:
            (x, y), (axis1, axis2), angle = cv2.fitEllipse(cnt)
            MA, ma = max(axis1, axis2), min(axis1, axis2)
            aspect_ratio = MA / ma if ma > 0 else 0
            eccentricity = np.sqrt(1 - (ma / MA) ** 2) if MA > 0 else 0
        else:
            MA = ma = aspect_ratio = eccentricity = 0

        # Convex hull-based features
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # RGB color features within the contour
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  # filled contour
        B, G, R = cv2.split(img)
        features_rgb = {
            "R_mean": np.mean(R[mask==255]),
            "R_std": np.std(R[mask==255]),
            "G_mean": np.mean(G[mask==255]),
            "G_std": np.std(G[mask==255]),
            "B_mean": np.mean(B[mask==255]),
            "B_std": np.std(B[mask==255])
        }

        # Combine shape and RGB features
        features = {
            "filename": file,
            "label": label,
            "area": area,
            "perimeter": perimeter,
            "major_axis": MA,
            "minor_axis": ma,
            "aspect_ratio": aspect_ratio,
            "eccentricity": eccentricity,
            "solidity": solidity,
            "roundness": roundness
        }
        features.update(features_rgb)
        
        all_features.append(features)
        pbar.update(1)

pbar.close()

# Save extracted features to CSV
df = pd.DataFrame(all_features)
df.to_csv(output_csv, index=False)
print(f"Features saved to {output_csv}")