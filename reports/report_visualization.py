import matplotlib.pyplot as plt
import numpy as np
import cv2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

root_dir = "special"

column_names = ["trajs", "heatmap", "pred_segs", "e2e_segs", "RN"]
column_display_names = ["", "Input Trajectories", "Density Maps", "Baseline (TR2RM)", "GraphWalker", "Ground Truth"]

row_names = ["Multi-Branches\nBetween\nJunctions", "Overpasses\nand\nUnderpasses", "Sparse\nInput\nTrajectory", "Self-loop\nRoad"]

# top, right, bottom, left percentage to crop
zoomin_pixels = [
    [0.24, 0, 0.26, 0],
    [0.24, 0, 0.26, 0],
    [0.35, 0, 0.15, 0],
    [0.00, 0, 0.50, 0]
]

def processHeatmap(img):
    # if a pixel is red, set all its neighbor pixels to red

    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    new_img = img.copy()

    for i in range(256):
        for j in range(256):
            if np.all(img[i, j] == [1, 0, 0]):
                row_range = (max(0, i-1), min(256, i+2))
                col_range = (max(0, j-1), min(256, j+2))
                new_img[row_range[0]:row_range[1], col_range[0]:col_range[1]] = [1, 0, 0]

    return new_img


def cropImage(img, zoomin_pixel):
    h, w = img.shape[:2]
    top, right, bottom, left = zoomin_pixel
    top = int(h * top)
    right = int(w * right)
    bottom = int(h * bottom)
    left = int(w * left)
    return img[top:h-bottom, left:w-right]


fig, axs = plt.subplots(4, 6, figsize=(11, 4), gridspec_kw={'width_ratios': [0.5, 1, 1, 1, 1, 1]})

font_size=13

for r in range(4):
    ax = axs[r, 0]
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"{row_names[r]}", fontsize=font_size, ha="center", va="center")
    for c in range(1, 6):
        ax = axs[r, c]
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        img = plt.imread(f"{root_dir}/{r}_{column_names[c-1]}.png")
        if c == 2:
            print(f"{root_dir}/{r}_{column_names[c-1]}.png")
            img = processHeatmap(img)
        img = cropImage(img, zoomin_pixels[r])
        ax.imshow(img)

for ax, col in zip(axs[0], column_display_names):
    ax.set_title(col, fontsize=font_size)

plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)

plt.tight_layout(pad=0.5)

plt.savefig(f"{root_dir}/visualization.pdf", dpi=300)

plt.show()