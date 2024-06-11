import os
import cv2

def classify_image(image_path, class_label):
    # add csv file to store the image path and class label
    with open("image_classification.csv", "a") as file:
        file.write(f"{image_path},{class_label}\n")

# Path to the folder
folder = "split_dataset"

# Iterate through the images in the folder
for image_file in os.listdir(folder):
    image_path = os.path.join(folder, image_file)

    # Display the image
    print(f"Displaying image: {image_path}")
    cv2.imshow("Image", cv2.imread(image_path))
    user_input = cv2.waitKey(0)

    # Ask the user for input (w, a, or d)
    #user_input = input(f"Classify image {image_file} as (w) himalaya, (a) not himalaya, or (d) garbage: ")

    # Classify the image based on user input
    if user_input == ord("w"):
        classify_image(image_path, "himalaya")
    elif user_input == ord("a"):
        classify_image(image_path, "not himalaya")
    elif user_input == ord("d"):
        pass
    else:
        print("Invalid input. Skipping the image.")

print("Image classification completed.")

import pandas as pd
import shutil

#df = pd.read_csv("image_classification.csv")
#for i, row in df.iterrows():
    #shutil.copyfile(os.path.join(row['image_path']), os.path.join("test1_data", "training", str(row['class_label']), row['image_path'].split("\\")[-1]))
