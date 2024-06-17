import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_feature_vector(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return descriptors

def compare_feature_vectors(query_vector, base_vectors):
    if query_vector is None or base_vectors is None:
        return 0
    # Initialize BF matcher
    bf = cv2.BFMatcher()

    # Match query vector with base vectors
    matches = bf.knnMatch(query_vector, base_vectors, k=2)

    # Filter matches based on Lowe's ratio test
    count=0
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
        else:
            continue
        if m.distance < 0.7 * n.distance:
            count+=1

    return count

def draw_bar_graph(matches):
    # Count the number of matches for each base vector
    match_counts = np.bincount(matches)

    # Plot the bar graph
    plt.plot(range(len(matches)), matches)
    plt.xlabel('Base Vector')
    plt.ylabel('Number of Matches')
    plt.show()

# # Load the query image
# query_image = cv2.imread('not_himalaya2.jpg')

# # Load the base data
# with open('base_data.bin', 'rb') as f:
#     base_data = pickle.load(f)

# # Get the feature vector from the query image
# query_vector = get_feature_vector(query_image)

# # Compare the query vector with the base vectors
# matches = []
# for base_vector in base_data:
#     match_count = compare_feature_vectors(query_vector, base_vector)
#     matches.append(match_count)

# # Draw the bar graph
# draw_bar_graph(matches)
# print(max(matches))