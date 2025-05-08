import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import shutil

# === Config ===
SOURCE_DIR = Path(r"C:\Users\Mink\OneDrive\Documents\GitHub\data\EMNIST\converted_png_fixed")
DEST_DIR = Path(r"C:\Users\Mink\OneDrive\Documents\GitHub\data\EMNIST\processed_112_augmented_cleaned")
UPSCALE_SIZE = (112, 112)
AUGMENT_PER_IMAGE = 2  # mỗi ảnh gốc sinh thêm mấy bản
BORDER_WIDTH = 6
BIN_THRESHOLD = 127  # ngưỡng nhị phân hóa

def mask_border(img, border=BORDER_WIDTH):
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[border:h-border, border:w-border] = 1
    return cv2.bitwise_and(img, img, mask=mask)

def binarize_image(img, threshold=BIN_THRESHOLD):
    return np.where(img > threshold, 255, 0).astype(np.uint8)

def augment_image(img):
    rows, cols = img.shape
    angle = random.uniform(-15, 15)
    tx = random.randint(-4, 4)
    ty = random.randint(-4, 4)
    
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    M[:, 2] += [tx, ty]
    
    return cv2.warpAffine(img, M, (cols, rows), borderValue=255)

def process_and_augment_all():
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True)

    class_dirs = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()])
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        dest_class = DEST_DIR / class_dir.name
        dest_class.mkdir(parents=True)

        for img_path in class_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img_up = cv2.resize(img, UPSCALE_SIZE, interpolation=cv2.INTER_LINEAR)
            img_bin = binarize_image(img_up)
            img_clean = mask_border(img_bin)

            # Save original binarized version
            base_name = img_path.stem
            cv2.imwrite(str(dest_class / f"{base_name}_orig.png"), img_clean)

            # Augmented versions
            for i in range(AUGMENT_PER_IMAGE):
                aug = augment_image(img_up)
                aug_bin = binarize_image(aug)
                aug_clean = mask_border(aug_bin)
                cv2.imwrite(str(dest_class / f"{base_name}_aug{i}.png"), aug_clean)

    print("Augment + Binarize done!")

if __name__ == "__main__":
    process_and_augment_all()
