import os
import matplotlib.pyplot as plt
import cv2

DATA_DIR = "C:/Users/iTech/Downloads/MRI/data/data_set1"
CATEGORIES = ["yes", "no"]

for category in CATEGORIES:
    folder = os.path.join(DATA_DIR, category)
    print(f"Category: {category}, Number of images: {len(os.listdir(folder))}")
    for img_name in os.listdir(folder)[:2]:
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, cmap="gray")
        plt.title(f"{category} - {img_name}")
        plt.axis("off")
        plt.show()
