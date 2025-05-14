import os
import random
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

#Config
DATA_DIR = Path(r"C:\Users\Mink\OneDrive\Documents\GitHub\data\EMNIST\processed_112_augmented_cleaned")
BATCH_SIZE = 64
EPOCHS = 20
SAMPLE_RATIO = 1 # 100% of the dataset
IMAGE_SIZE = 112
NUM_CLASSES = 36  # 0–9 + a–z

#Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

#Dataset & Sample
full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)
total_len = len(full_dataset)
sample_len = int(SAMPLE_RATIO * total_len)
indices = random.sample(range(total_len), sample_len)
subset = Subset(full_dataset, indices)

loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

#Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (IMAGE_SIZE//4) * (IMAGE_SIZE//4), 256), nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x): return self.net(x)

#Train
def train():
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "simple_cnn_emnist.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()

    # === Save model manually after interrupt ===
    torch.save(model.state_dict(), "simple_cnn_epoch4_partial.pth")
    print("Model saved as simple_cnn_epoch4_partial.pth") 