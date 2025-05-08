import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# CONFIG
DATA_DIR = Path(r"C:\Users\Mink\OneDrive\Documents\GitHub\data\EMNIST\processed_112_augmented_cleaned")
IMAGE_SIZE = 112
BATCH_SIZE = 64
SAMPLE_SIZE = 2000
NUM_CLASSES = 36
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = "eval_report.csv"

# TRANSFORM
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# DATASET 
full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)
class_names = full_dataset.classes
indices = random.sample(range(len(full_dataset)), SAMPLE_SIZE)
subset = Subset(full_dataset, indices)
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

# MODELS 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (IMAGE_SIZE // 4) ** 2, 256), nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )
    def forward(self, x): return self.net(x)

class MiniCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.rnn = nn.LSTM(input_size=32 * (IMAGE_SIZE // 4), hidden_size=128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)  # [B, W/4, 32, H/4]
        x = x.flatten(2)
        rnn_out, _ = self.rnn(x)
        last_step = rnn_out[:, -1, :]
        return self.classifier(last_step)

# EVALUATION FUNCTION
def evaluate_model(model, loader, device, name, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Evaluating {name}"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    show_report_and_matrix(all_labels, all_preds, name)
    save_report_csv(name, all_labels, all_preds)

def show_report_and_matrix(y_true, y_pred, model_name):
    print(f"\n=== Report for {model_name} ===")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, annot=False, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_{model_name}.png")
    print(f"Confusion matrix saved to conf_matrix_{model_name}.png")

def save_report_csv(model_name, y_true, y_pred, output_path=CSV_PATH):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    df = pd.DataFrame(report_dict).transpose()
    df["model"] = model_name
    df["accuracy"] = acc

    write_header = not os.path.exists(output_path)
    df.to_csv(output_path, mode="a", header=write_header, index=True)
    print(f"Evaluation results for {model_name} saved to {output_path}")

# MAIN
def main():
    models_to_eval = [
        ("SimpleCNN", SimpleCNN(), "simple_cnn_emnist.pth")
    ]

    for name, model, path in models_to_eval:
        if os.path.exists(path):
            evaluate_model(model, loader, DEVICE, name, path)
        else:
            print(f"Model file {path} not found, skipping {name}.")

if __name__ == "__main__":
    main()
    print("Evaluation completed.")