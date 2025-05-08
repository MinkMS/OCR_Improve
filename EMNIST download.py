import os
import requests
import zipfile

# Đường dẫn thư mục lưu
data_dir = 'C:/Users/Mink/OneDrive/Documents/GitHub/data/EMNIST/raw'
os.makedirs(data_dir, exist_ok=True)

# Link GitHub Releases
url = 'https://github.com/hoangtrung247/emnist-assets/releases/download/v1.0/emnist-gzip.zip'
output_zip = os.path.join(data_dir, 'gzip.zip')

# Tải bằng stream
print("Đang tải EMNIST từ GitHub...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(output_zip, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Giải nén
print("Đang giải nén...")
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

print("Hoàn tất")
