import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from preprocess import ImagePreprocessor
from mysterydevice_model import NeuralNetworkInference
import gc

preprocessor = ImagePreprocessor()

input_nodes = 28 * 28
output_nodes = 10
total_models = 12
hidden_layer_sizes = [[110 + (i * 3)] for i in range(total_models)]

def load_image(filename):
    return preprocessor.process(filename)


def classify_image(image, threshold=0.85):
    all_probs = []
    for i in range(total_models):
        m = NeuralNetworkInference(f'weights_{i}.npz')
        _, probs = m.predict(image)
        all_probs.append(probs.flatten().copy())
        del m
        gc.collect()

    probs = np.mean(all_probs, axis=0)

    best_indices = np.argsort(probs)
    highest_idx = best_indices[-1]
    second_idx = best_indices[-2]

    highest_prob = probs[highest_idx]
    second_prob = probs[second_idx]

    margin = highest_prob - second_prob

    if highest_prob < threshold or margin < 0.8:
        primary_pred = "niet herkend"
    else:
        primary_pred = str(highest_idx)

    guess_if_forced = str(highest_idx)

    return primary_pred, highest_prob, guess_if_forced, second_idx, second_prob


class DrawingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw a digit")

        self.pil_image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.pil_image)

        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black", cursor="crosshair")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()
        tk.Button(btn_frame, text="Classify", command=self.classify).pack(side="left")
        tk.Button(btn_frame, text="Clear",    command=self.clear).pack(side="left")

        self.label = tk.Label(self.root, text="Draw a digit, then click Classify")
        self.label.pack()

        self.preview_label = tk.Label(self.root)
        self.preview_label.pack()

        self.last_x = None
        self.last_y = None

    def paint(self, event):
        brush_size = 10
        x, y = event.x, event.y

        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    fill="white", width=brush_size * 2,
                                    capstyle=tk.ROUND, smooth=True)
            self.draw.ellipse([x - brush_size, y - brush_size, x + brush_size, y + brush_size], fill="white")
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill="white", width=brush_size * 2)
        else:
            self.canvas.create_oval(x - brush_size, y - brush_size, x + brush_size, y + brush_size, fill="white", outline="")
            self.draw.ellipse([x - brush_size, y - brush_size, x + brush_size, y + brush_size], fill="white")

        self.last_x, self.last_y = x, y

    def reset_last(self, event):
        self.last_x = None
        self.last_y = None

    def classify(self):
        small = self.pil_image.resize((28, 28), Image.LANCZOS)
        image = preprocessor.process_array(np.array(small, dtype=np.uint8))

        pred, prob, forced, second_idx, second_prob = classify_image(image)
        self.label.config(
            text=f"Prediction: {pred}  ({prob:.1%})  |  Runner-up: {second_idx} ({second_prob:.1%})"
        )

        preview_arr = (image.reshape(28, 28) * 255).astype(np.uint8)
        preview_img = Image.fromarray(preview_arr, mode="L").resize((140, 140), Image.NEAREST)
        self.preview_image = ImageTk.PhotoImage(preview_img)
        self.preview_label.config(image=self.preview_image)

    def clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.pil_image)
        self.label.config(text="Draw a digit, then click Classify")
        self.preview_label.config(image="")
        self.preview_image = None

    def run(self):
        self.root.mainloop()
