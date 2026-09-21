import os
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from preprocess import preprocess_drawn_image
from model import DigitCNN

class DigitRecognizerApp:
    def __init__(self, root, model=None, device="cpu"):
        self.root = root
        self.root.title("Handwritten Digit Recognizer - Member 1")
        self.root.resizable(False, False)
        
        self.model = model
        self.device = device

        self.canvas_size = 280
        self.brush_size = 18

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self._build_ui()

    def _build_ui(self):
        title_font = tkfont.Font(family="Helvetica", size=14, weight="bold")
        result_font = tkfont.Font(family="Helvetica", size=24, weight="bold")

        header = tk.Label(self.root, text="Draw a digit (0-9) inside the box", font=title_font)
        header.grid(row=0, column=0, columnspan=2, pady=(10, 5))

        self.canvas = tk.Canvas(
            self.root, 
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg="white", 
            cursor="cross"
        )
        self.canvas.grid(row=1, column=0, padx=15, pady=10)
        self.canvas.bind("<B1-Motion>", self._paint)

        info_frame = tk.Frame(self.root)
        info_frame.grid(row=1, column=1, padx=15, pady=10, sticky="n")

        tk.Label(info_frame, text="Prediction:", font=title_font).pack(anchor="w")
        self.lbl_prediction = tk.Label(info_frame, text="-", font=result_font, fg="#1a73e8")
        self.lbl_prediction.pack(anchor="w", pady=(0, 10))

        tk.Label(info_frame, text="Confidence:", font=title_font).pack(anchor="w")
        self.lbl_confidence = tk.Label(info_frame, text="-", font=tkfont.Font(size=12))
        self.lbl_confidence.pack(anchor="w")

        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(5, 15))

        btn_clear = tk.Button(btn_frame, text="Clear Canvas", width=12, command=self.clear_canvas)
        btn_clear.pack(side="left", padx=10)

        btn_predict = tk.Button(btn_frame, text="Predict", width=12, bg="#4CAF50", fg="white", command=self.predict_digit)
        btn_predict.pack(side="left", padx=10)

    def _paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_prediction.config(text="-")
        self.lbl_confidence.config(text="-")

    def predict_digit(self):
        raw_array = np.array(self.image)
        processed_tensor = preprocess_drawn_image(raw_array)

        if self.model is None:
            self.lbl_prediction.config(text="Demo")
            self.lbl_confidence.config(text="No weights loaded")
            return

        self.model.eval()
        with torch.no_grad():
            tensor = processed_tensor.to(self.device)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            top_class = int(np.argmax(probs))
            confidence = probs[top_class] * 100.0

            self.lbl_prediction.config(text=str(top_class))
            self.lbl_confidence.config(text=f"{confidence:.2f}%")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "best_mnist_cnn.pth"

    if os.path.exists(weights_path):
        model = DigitCNN().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights successfully from '{weights_path}'")
    else:
        print(f"Warning: '{weights_path}' not found. Run train.py first!")
        model = None

    root = tk.Tk()
    app = DigitRecognizerApp(root, model=model, device=device)
    root.mainloop()