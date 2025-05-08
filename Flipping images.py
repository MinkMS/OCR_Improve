import os
import numpy as np
from PIL import Image

# Đường dẫn thư mục chứa ảnh đã convert
base_dir = r"C:/Users/Mink/OneDrive/Documents/GitHub/data/EMNIST/converted_png"
output_dir = r"C:/Users/Mink/OneDrive/Documents/GitHub/data/EMNIST/converted_png_fixed"

# Mở file ánh xạ label -> ký tự (tạo map hoa thường về cùng lớp)
mapping_path = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\EMNIST\emnist-byclass-mapping.txt"
label_map = {}

# Load mapping label -> ký tự
with open(mapping_path, 'r') as f:
    for line in f:
        label_id, ascii_code = map(int, line.strip().split())
        label_map[label_id] = chr(ascii_code)

# Tạo thư mục output_fixed nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# Duyệt qua tất cả thư mục con (class) sử dụng os.walk để tránh lỗi
for folder_name, subfolders, filenames in os.walk(base_dir):
    if folder_name == base_dir:  # Nếu thư mục gốc
        continue
    
    # Tạo thư mục cho từng class đã sửa (class_name = tên của thư mục)
    class_name = label_map.get(str(os.path.basename(folder_name)), os.path.basename(folder_name))
    class_folder = os.path.join(output_dir, class_name)
    os.makedirs(class_folder, exist_ok=True)
    
    # Duyệt qua tất cả file ảnh trong thư mục này
    for filename in filenames:
        file_path = os.path.join(folder_name, filename)
        if file_path.endswith('.png'):  # Chỉ xử lý ảnh .png
            try:
                # Mở ảnh và chỉnh sửa
                img = Image.open(file_path)
                img = np.array(img)  # chuyển ảnh thành numpy array để xử lý

                # Lật ảnh (flipping)
                fixed_img = np.fliplr(img)  # Lật ngược theo trục ngang

                # Lưu ảnh vào thư mục class đã tạo
                new_file_path = os.path.join(class_folder, filename)
                fixed_img_pil = Image.fromarray(fixed_img)
                fixed_img_pil.save(new_file_path)  # Ghi ảnh đã lật vào thư mục đúng class

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue  # Nếu có lỗi với ảnh thì bỏ qua và xử lý ảnh khác

print("Đã lật lại tất cả ảnh và chuyển vào thư mục theo class.")
