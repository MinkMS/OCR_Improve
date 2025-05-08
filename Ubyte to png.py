import os
import struct
import numpy as np
from PIL import Image

# ==== Đường dẫn ====
data_dir = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\EMNIST"
images_path = os.path.join(data_dir, 'emnist-byclass-train-images-idx3-ubyte')
labels_path = os.path.join(data_dir, 'emnist-byclass-train-labels-idx1-ubyte')
mapping_path = os.path.join(data_dir, 'emnist-byclass-mapping.txt')
output_dir = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\EMNIST\converted_png"

os.makedirs(output_dir, exist_ok=True)

# ==== Hàm load ảnh ====
def load_images(filepath):
    with open(filepath, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape((num, rows, cols))

# ==== Hàm load label ====
def load_labels(filepath):
    with open(filepath, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# ==== Hàm mapping label → ký tự ====
def load_label_mapping(filepath):
    label_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            label_id, ascii_code = map(int, line.strip().split())
            label_map[label_id] = chr(ascii_code)
    return label_map

# ==== Fix chiều ảnh EMNIST ====
def fix_orientation(image):
    return np.transpose(np.flip(image, axis=0))  # FIX CHUẨN

# ==== Load dữ liệu ====
print("Đang load dữ liệu...")
images = load_images(images_path)
labels = load_labels(labels_path)
label_map = load_label_mapping(mapping_path)

assert len(images) == len(labels), "Số lượng ảnh và nhãn không khớp!"

# ==== Convert và lưu ====
print("Đang chuyển ảnh và lưu về PNG...")
counter = {}

for idx, (img, lbl) in enumerate(zip(images, labels)):
    char = label_map[lbl]

    # Đường dẫn thư mục class
    class_dir = os.path.join(output_dir, char)
    os.makedirs(class_dir, exist_ok=True)

    # Đổi chiều ảnh về chuẩn
    fixed_img = fix_orientation(img)

    # Chuyển sang PIL và lưu
    pil_img = Image.fromarray(fixed_img)
    filename = f"{counter.get(char, 0):05}.png"
    pil_img.save(os.path.join(class_dir, filename))

    counter[char] = counter.get(char, 0) + 1

    # Log tiến độ
    if idx % 10000 == 0:
        print(f"Đã xử lý {idx}/{len(images)} ảnh...")

print("Ảnh đã được lưu về:", output_dir)
print("Tổng số ảnh:", sum(counter.values()))