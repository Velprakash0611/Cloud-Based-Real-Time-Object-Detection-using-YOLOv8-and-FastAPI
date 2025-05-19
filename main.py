import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
import base64
import cv2
import numpy as np
import io

API_URL = "https://cloud-based-real-time-object-detection.onrender.com/predict"

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def detect_objects():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    encoded_img = encode_image(file_path)

    try:
        response = requests.post(API_URL, json={"image": encoded_img}, timeout=15)
        result = response.json()

        if not result.get("success"):
            messagebox.showerror("Error", result.get("error"))
            return

        image = cv2.imread(file_path)
        image = cv2.resize(image, (640, 480))

        for pred in result["predictions"]:
            x1, y1, x2, y2 = pred["xyxy"]
            label = pred["label"]
            conf = pred["confidence"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(img_pil)

        image_label.config(image=img_tk)
        image_label.image = img_tk

    except requests.exceptions.RequestException as e:
        messagebox.showerror("Request Error", str(e))

# UI Setup
root = tk.Tk()
root.title("YOLOv8 Object Detection")
root.geometry("700x600")

btn_upload = tk.Button(root, text="Upload and Detect Image", command=detect_objects, font=("Arial", 14))
btn_upload.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

root.mainloop()
