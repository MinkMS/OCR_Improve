import numpy as np
import matplotlib.pyplot as plt
import struct
import os

# ==== Đường dẫn file gốc ====
data_dir = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\EMNIST"
images_path = os.path.join(data_dir, 'emnist-byclass-train-images-idx3-ubyte')
labels_path = os.path.join(data_dir, 'emnist-byclass-train-labels-idx1-ubyte')
mapping_path = os.path.join(data_dir, 'emnist-byclass-mapping.txt')

# ==== Hàm đọc ảnh ====
def load_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape((num, rows, cols))

# ==== Hàm đọc label ====
def load_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# ==== Hàm đọc mapping label → ASCII char ====
def load_label_mapping(mapping_file):
    label_map = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            label_id, ascii_code = map(int, line.strip().split())
            label_map[label_id] = chr(ascii_code)
    return label_map

# ==== Chuẩn hóa chiều ảnh EMNIST ====
def fix_orientation(image):
    return np.transpose(np.flip(image, axis=0))

# ==== Load dữ liệu ====
images = load_images(images_path)
labels = load_labels(labels_path)
label_map = load_label_mapping(mapping_path)

print(f"Loaded {len(images)} images.")

# ==== Hiển thị vài ảnh đầu ====
plt.figure(figsize=(10, 4))
for i in range(10):
    img = fix_orientation(images[i])
    label = labels[i]
    char = label_map[label]

    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {char}")
    plt.axis('off')

plt.tight_layout()
plt.show()
