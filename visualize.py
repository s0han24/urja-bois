import get_data
import cv2
import pickle
import matplotlib.pyplot as plt

# Load the query image
def load_image_vector(image_path):
    query_image = cv2.imread(image_path)
    query_vector = get_data.get_feature_vector(query_image)
    return query_vector

# Load the base data
def load_base_data(file_path):
    with open(file_path, 'rb') as f:
        base_data = pickle.load(f)
    return base_data

# get test images from image_classification_test.csv file
def get_test_images():
    test_images = []
    idx = 0
    with open('image_classification_test.csv', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if idx == 200:
                break
            test_images.append((line.strip().split(',')[0], line.strip().split(',')[1]))
            idx += 1
    return test_images

base_data = load_base_data('base_data.bin')

split_data = {'himalaya':[], 'not himalaya':[]}

for image in get_test_images():
    query_vector = load_image_vector(image[0])
    sum_match = 0
    max_match = 0
    for base_vector in base_data:
        match_count = get_data.compare_feature_vectors(query_vector, base_vector)
        sum_match += match_count
        if match_count > max_match:
            max_match = match_count
    split_data[image[1]].append(max_match)

plt.scatter(range(len(split_data['himalaya'])), split_data['himalaya'], color='blue', label='himalaya')
plt.scatter(range(len(split_data['not himalaya'])), split_data['not himalaya'], color='red', label='not himalaya')
plt.xlabel('Image')
plt.ylabel('Number of Matches')
plt.legend()
plt.show()
        