import cv2
import matplotlib.pyplot as plt
import os

# Load the image
base_image = cv2.imread('test1\\1.png')

# Convert the image to grayscale
gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

# blur the image
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Initialize the SIFT feature detector
sift = cv2.SIFT_create()

# Detect and compute the keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw the keypoints on the image
annotated_image = cv2.drawKeypoints(gray, keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the annotated image
cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

fig, axs = plt.subplots(4, 4)

imgs = []
no_of_matches = {}
folder = "test1"

for id, img_file in enumerate(os.listdir(folder)):
    img = cv2.imread(os.path.join(folder, img_file))
    imgs.append(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    keypoints2, descriptors2 = sift.detectAndCompute(gray, None)
    if descriptors2 is None:
        no_of_matches[img_file] = 0
        continue
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors2, k=2)
    count = 0
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
        else:
            continue
        if m.distance < 0.75 * n.distance:
            count += 1
    no_of_matches[img_file] = count/len(matches)

def return_difference(image):
    return no_of_matches[image]
imgs.sort(key = return_difference, reverse = True)

n = len(imgs)
for i in range(16):
    print(imgs[i], no_of_matches[imgs[i]])
    axs[i // 4, i % 4].imshow(cv2.cvtColor(cv2.imread(os.path.join(folder, imgs[i])), cv2.COLOR_BGR2RGB))
    axs[i // 4, i % 4].set_title(no_of_matches[imgs[i]])
    axs[i // 4, i % 4].axis("off")
plt.show()    