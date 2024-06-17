import cv2
import numpy as np
import os
import pickle

def extract_features(image_dir, output_file):
    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Initialize an empty list to store the feature vectors
    feature_vectors = []

    # Iterate over all the images in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Blur the image
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detect keypoints and compute descriptors
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            # Add the descriptors to the feature vectors list
            feature_vectors.append(descriptors)

    # Save the feature vectors to a binary file
    with open(output_file, 'wb') as f:
        pickle.dump(feature_vectors, f)

# Specify the directory containing the images
image_dir = 'base_data'

# Specify the output file path
output_file = 'base_data.bin'

# Call the function to extract features and store them in a binary file
extract_features(image_dir, output_file)