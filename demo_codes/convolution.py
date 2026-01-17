import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons


#python3 demo_codes/convolution.py demo_codes/img/cat.png 
# -----------------------------
# Argumenty z terminala
# -----------------------------
parser = argparse.ArgumentParser(description="Demo konwolucji 2D (CNN / YOLO)")
parser.add_argument("image_path", help="Ścieżka do obrazu wejściowego")
args = parser.parse_args()

# -----------------------------
# Wczytanie obrazu
# -----------------------------
img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Nie można wczytać obrazu: {args.image_path}")

img = cv2.resize(img, (256, 256))

# -----------------------------
# Kernels
# -----------------------------
def get_kernel(kernel_type, ksize, scale):
    if kernel_type == "blur":
        kernel = np.ones((ksize, ksize)) / (ksize * ksize)
    elif kernel_type == "edges":
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])
    elif kernel_type == "sharpen":
        kernel = np.array([
            [0, -1,  0],
            [-1, 5, -1],
            [0, -1,  0]
        ])
    else:
        kernel = np.ones((3, 3))

    return kernel * scale

# -----------------------------
# Konwolucja
# -----------------------------
def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# -----------------------------
# Matplotlib UI
# -----------------------------
fig, axes = plt.subplots(1, 2)
plt.subplots_adjust(bottom=0.35)

ax_orig, ax_conv = axes
ax_orig.set_title("Oryginał")
ax_conv.set_title("Po konwolucji")

im_orig = ax_orig.imshow(img, cmap="gray")
kernel = get_kernel("blur", 3, 1.0)
im_conv = ax_conv.imshow(convolve(img, kernel), cmap="gray")

# --- Slidery ---
ax_ksize = plt.axes([0.2, 0.25, 0.6, 0.03])
ax_scale = plt.axes([0.2, 0.2, 0.6, 0.03])

s_ksize = Slider(ax_ksize, "Kernel size", 1, 9, valinit=3, valstep=2)
s_scale = Slider(ax_scale, "Scale", 0.1, 3.0, valinit=1.0)

# --- Radio buttons ---
ax_radio = plt.axes([0.02, 0.5, 0.15, 0.2])
radio = RadioButtons(ax_radio, ("blur", "edges", "sharpen"))

# -----------------------------
# Update callback
# -----------------------------
def update(val):
    ksize = int(s_ksize.val)
    scale = s_scale.val
    ktype = radio.value_selected

    kernel = get_kernel(ktype, ksize, scale)
    conv = convolve(img, kernel)

    im_conv.set_data(conv)
    fig.canvas.draw_idle()

s_ksize.on_changed(update)
s_scale.on_changed(update)
radio.on_clicked(update)

plt.show()
