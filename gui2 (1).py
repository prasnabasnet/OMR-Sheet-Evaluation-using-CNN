import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import os
import sys

# Path for PyInstaller compatibility
BASE_DIR = getattr(sys, '_MEIPASS', os.path.abspath("."))
model_path = os.path.join(BASE_DIR, "best_cnn_model.keras")

# Load model (compile=False for speed, safe_mode=True for safety in .exe)
model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
true_labels = ['filled', 'invalid', 'unfilled']

class OMREvaluatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(" OMR Sheet Evaluator")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f4f8")

        top_frame = tk.Frame(root, bg="#f0f4f8")
        top_frame.pack(pady=10)

        self.upload_button = tk.Button(top_frame, text=" Upload OMR Sheet", font=("Arial", 12, "bold"),
                                       bg="#4CAF50", fg="white", command=self.upload_image)
        self.upload_button.pack()

        content_frame = tk.Frame(root, bg="#f0f4f8")
        content_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(content_frame, width=400, height=500, bg="white", bd=2, relief="groove")
        self.canvas.grid(row=0, column=0, padx=20)

        self.tree = ttk.Treeview(content_frame, columns=("Q", "Answer"), show="headings", height=20)
        self.tree.heading("Q", text="Question")
        self.tree.heading("Answer", text="Detected")
        self.tree.column("Q", anchor=tk.CENTER, width=100)
        self.tree.column("Answer", anchor=tk.CENTER, width=120)
        self.tree.grid(row=0, column=1, padx=20, sticky="n")

        self.summary_label = tk.Label(root, text="", font=("Arial", 13, "bold"), bg="#f0f4f8", fg="#333")
        self.summary_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return
        try:
            final_answers, score_summary, annotated_image = self.evaluate(file_path)
            self.show_image(annotated_image)
            self.display_answers(final_answers)
            self.summary_label.config(text=score_summary)
        except Exception as e:
            messagebox.showerror(" Error", str(e))

    def evaluate(self, path):
        img = cv2.imread(path)
        if img is None:
            raise Exception("Failed to load image.")

        # Crop ROI
        x1, y1, x2, y2 = 11, 17, 1094, 1473
        roi = img[y1:y2, x1:x2].copy()

        # Preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def is_valid_bubble(cnt):
            area = cv2.contourArea(cnt)
            if area < 800 or area > 3500: return False
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / float(h)
            if ar < 0.4 or ar > 2.5: return False
            if h > 1.5 * w or w > 1.5 * h: return False
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: return False
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            return circularity >= 0.3

        bubble_contours = [c for c in contours if is_valid_bubble(c)]
        boxes = [cv2.boundingRect(c) for c in bubble_contours]
        centers = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]
        ys = np.array([[cy] for (cx, cy) in centers])
        kmeans = KMeans(n_clusters=10, random_state=42).fit(ys)
        labels = kmeans.labels_

        rows = [[] for _ in range(10)]
        for idx, lbl in enumerate(labels):
            rows[lbl].append((centers[idx][0], bubble_contours[idx]))
        rows.sort(key=lambda r: np.mean([cv2.boundingRect(c[1])[1] for c in r]))
        for i in range(10):
            rows[i].sort(key=lambda x: x[0])
        sorted_bubbles = [c for row in rows for _, c in row]

        # Batch Prediction
        bubble_imgs = []
        bubble_data = []
        for cnt in sorted_bubbles:
            x, y, w, h = cv2.boundingRect(cnt)
            bubble_data.append((cnt, x, y, w, h))
            bubble_img = roi[y:y + h, x:x + w]
            bubble_resized = cv2.resize(bubble_img, (64, 64))
            bubble_array = img_to_array(bubble_resized) / 255.0
            bubble_imgs.append(bubble_array)

        bubble_imgs = np.array(bubble_imgs)
        predictions = model.predict(bubble_imgs, verbose=0)
        bubble_predictions = [true_labels[np.argmax(p)] for p in predictions]

        # Annotate
        for i, (cnt, x, y, w, h) in enumerate(bubble_data):
            label = bubble_predictions[i]
            color = (0, 255, 0) if label == "filled" else (255, 0, 0) if label == "unfilled" else (0, 0, 255)
            cv2.rectangle(roi, (x, y), (x + w, y + h), color, 2)
            cv2.putText(roi, label[0].upper(), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Extract answers
        final_answers = ['Invalid'] * 20
        for i, row in enumerate(rows):
            if len(row) < 8: continue
            left4 = bubble_predictions[i * 8: i * 8 + 4]
            right4 = bubble_predictions[i * 8 + 4: i * 8 + 8]
            if left4.count('filled') == 1 and left4.count('unfilled') == 3:
                final_answers[i] = chr(65 + left4.index('filled'))
            elif left4.count('unfilled') == 4:
                final_answers[i] = "Unfilled"
            if right4.count('filled') == 1 and right4.count('unfilled') == 3:
                final_answers[i + 10] = chr(65 + right4.index('filled'))
            elif right4.count('unfilled') == 4:
                final_answers[i + 10] = "Unfilled"

        # Load key and calculate score
        key_path = os.path.join(BASE_DIR, "answer_key.txt")
        if not os.path.exists(key_path):
            raise Exception("Missing 'answer_key.txt'")
        with open(key_path, "r") as f:
            key = f.read().strip().upper()
        if len(key) != 20 or not all(c in 'ABCD' for c in key):
            raise Exception("Answer key must be 20 characters (A–D).")

        correct, wrong, unfilled, invalid = 0, 0, 0, 0
        for i in range(20):
            pred = final_answers[i]
            if pred == key[i]:
                correct += 1
            elif pred == "Unfilled":
                unfilled += 1
            elif pred == "Invalid":
                invalid += 1
            else:
                wrong += 1

        score = correct * 1 - wrong * 0.25
        summary = f" Correct: {correct}     Unfilled: {unfilled}     Invalid: {invalid}     Score: {score:.2f} / 20.00"
        return final_answers, summary, roi

    def show_image(self, roi_cv2):
        roi_resized = cv2.resize(roi_cv2, (400, 500))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(roi_rgb)
        im_tk = ImageTk.PhotoImage(im_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=im_tk)
        self.canvas.image = im_tk

    def display_answers(self, final_answers):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, ans in enumerate(final_answers):
            self.tree.insert("", tk.END, values=(f"Q{i+1:02}", ans))

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = OMREvaluatorGUI(root)
    root.mainloop()
