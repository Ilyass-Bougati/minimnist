import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import random
from sys import argv

# Reading the ideal numbers list
print("Reading the ideal numbers")
ideal = []
with open("images/images_ideals.csv", encoding="utf-8-sig") as file:
    filereader = csv.reader(file)
    for row in filereader:
        number = []
        for num in row:
            number.append(int(num))
        ideal.append(np.array(number))

# Reshaping the vectors to images
ideal_images = []
for i in range(len(ideal)):
    ideal_images.append({
        "img": np.array(ideal[i][:72]).reshape((12, 6)).astype(np.uint8),
        "label": ideal[i][-1]
    })

# This function is to prevent duplicates
def exists(pool: list, image) -> bool:
    for candidate in pool:
        match = True
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if candidate[i][j] != image[i][j]:
                    global counter
                    match = False
                    break
            if not match:
                break
        if match:
            return True
    return False

def add(pool: list, image: list) -> None:
    if (not exists(pool, image)):
        pool.append(image)

# Binarizing the images
def binarize(img: list) -> list:
    res = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i][j] = 1 if img[i][j] > 0.1 else 0
    return res

# applying a non linear warp to the images
def warp(img: list, angle: float,center_x: float, center_y: float) -> list:
    height, width = img.shape[:2]

    # Define the maximum radius for which the warp is applied (e.g., 80% of the minimum half-dimension)
    max_radius = min(center_x, center_y) * 0.5
    # Define the maximum rotation angle (in radians) at the very center
    max_angle = np.radians(angle)

    # Create a grid of (x, y) coordinates
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    # Shift coordinates so that the center is at (0, 0)
    x = map_x - center_x
    y = map_y - center_y
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Define the rotational offset: full rotation at r=0, tapering to 0 at r>=max_radius.
    angle_offset = np.zeros_like(r)
    mask = r < max_radius
    angle_offset[mask] = max_angle * (1 - r[mask] / max_radius)

    # Compute the new angle by adding the offset
    theta_new = theta + angle_offset

    # Convert back to Cartesian coordinates
    map_x_new = center_x + r * np.cos(theta_new)
    map_y_new = center_y + r * np.sin(theta_new)

    # Optionally, leave the pixels outside the warp radius unchanged:
    map_x_new[~mask] = map_x[~mask]
    map_y_new[~mask] = map_y[~mask]

    # Remap the image using the computed coordinate maps.
    warped = cv2.remap(img, map_x_new.astype(np.float32), map_y_new.astype(np.float32), cv2.INTER_LINEAR)
    return warped


# Generating the dataset, number by number
# Creating a list of kernels
print("Generating new images")
kernels = []
for i in range(1, 3):
    for j in range(1, 3):
        kernels.append(np.ones((i,j),np.uint8))

final_dataset = []
for i in range(len(ideal_images)):
    number_images = [ideal_images[i]["img"]]

    # Erroding and dilating the number
    for kernel in tqdm(kernels):
        add(number_images, cv2.erode(number_images[0], kernel, iterations=1))
        add(number_images, cv2.dilate(number_images[0], kernel, iterations=1))
        

    # for d in tqdm(range(len(number_images))):
    #     for j in range(1, 6):
    #         for k in range(7):
    #             for l in range(13):
    #                 add(number_images, binarize(warp(number_images[d], j * 10, k, l)))


    for d in tqdm(range(len(number_images))):
        for j in range(12):
            for k in range(6):
                img = np.copy(number_images[d])
                img[j][k] = (0 if ideal_images[i]["img"][j][k] == 1 else 1)
                add(number_images, (img))

    for image_index in range(len(number_images)):
        final_dataset.append(np.append(np.array(number_images[image_index]).flatten(), ideal_images[i]["label"]))

# removing the images that are too filled
new_dataset = []
for image in final_dataset:
    if float(np.sum(image[:-1])/len(image[:-1])) < 0.8:
        new_dataset.append(image)

# Plotting
try:
    if argv[1] == "plot":
        # Selecting a random set of pictures and showing them
        print("Plotting...")
        for i in range(60):
            plt.subplot(6, 10, 1 + i)
            # plt.subplots_adjust(bottom=0.9, right=1, top=2.5)
            random_img = random.choice(new_dataset)
            # random_img = final_dataset[i]
            random_img_2d = np.zeros((12, 6))
            for j in range(12):
                for k in range(6):
                    random_img_2d[j][k] = random_img[j * 6 + k]
            plt.axis("off")
            plt.imshow(random_img_2d, "gray_r")
            plt.tight_layout()
            plt.title(random_img[-1])
        plt.show()
except IndexError:
    print("Plotting skipped.")


print("Writting images...")
with open("images/dataset.csv", "a", newline="") as file:
    spamreader = csv.writer(file)
    for row in new_dataset:
        spamreader.writerow(row)


print(len(new_dataset))