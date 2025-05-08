import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (112//4) * (112//4), 256), nn.ReLU(),  # Adjusted for 112x112 input
            nn.Linear(256, 36)  # 36 output classes (0-9 + a-z)
        )

    def forward(self, x):
        return self.net(x)

# Load the model
model_path = r"C:\Users\Mink\OneDrive\Documents\GitHub\simple_cnn_emnist.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN()

# Try loading the checkpoint and handle mismatches
try:
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Preprocessing function for input image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((112, 112)),  # Match the training image size (112x112)
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalization as used during training
])

# Tkinter GUI for drawing the symbol
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Symbol Recognition")

        # Set a larger window size
        self.root.geometry("800x800")  # Width x Height (in pixels)

        # Canvas for drawing (increased size for more drawing space)
        self.canvas_width = 400  # Increased canvas width for more space
        self.canvas_height = 400  # Increased canvas height for more space
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(pady=20)

        # Buttons with increased font size
        self.btn_predict = tk.Button(self.root, text="Predict", command=self.predict, font=("Arial", 16))
        self.btn_predict.pack(side=tk.LEFT, padx=20)

        self.btn_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas, font=("Arial", 16))
        self.btn_clear.pack(side=tk.RIGHT, padx=20)

        # PIL image for drawing (adjusted size)
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        r = 10  # Radius of the circle - Brush size
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="white")

    def predict(self):
        # Convert canvas image to model input
        img = self.image.convert("L")  # Convert to grayscale
        img_resized = transform(img).unsqueeze(0)  # Apply preprocessing

        # Model inference
        with torch.no_grad():
            output = model(img_resized)
            prediction = torch.argmax(output, dim=1).item()

        # Display result
        if 0 <= prediction <= 9:
            predicted_symbol = str(prediction)  # Digits 0-9
        else:
            predicted_symbol = chr(prediction + 87)  # Convert 10-35 to 'a' to 'z'

        messagebox.showinfo("Prediction", f"Predicted Symbol: {predicted_symbol}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
