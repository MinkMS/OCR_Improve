import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Định nghĩa transform cho ảnh
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize ảnh về kích thước đồng nhất
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dùng ImageFolder để load ảnh
train_dataset = datasets.ImageFolder(root='C:/Users/Mink/OneDrive/Documents/GitHub/data/EMNIST/converted_png', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Kiểm tra vài thông tin
print(f"Dataset size: {len(train_dataset)} images")
print(f"Number of classes: {len(train_dataset.classes)}")
