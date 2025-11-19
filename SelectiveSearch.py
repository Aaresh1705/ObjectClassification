import cv2
import os
import random 
import numpy as np
import sys

def edgeboxes(im):
    model = 'model.yml.gz'
    # im = cv2.imread(sys.argv[2])

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(30)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    for b in boxes:
        x, y, w, h = b
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

    return im


folder = '/dtu/datasets1/02516/potholes/images'

files = [f for f in os.listdir(folder) if f.endswith('.png')]
img_name = random.choice(files)
img_path = os.path.join(folder, img_name)

img = cv2.imread(img_path)

cv2.imwrite("testimage.png", img)

# --- 2. Define Parameters ---
# spatialRadius (sp): Defines the neighborhood size for spatial grouping. 
# colorRadius (sr): Defines the range of color/intensity considered the same.
# maxLevel: Pyramid level for performance (usually 1 or 2).
spatial_radius = 20
color_radius = 45 
max_level = 1

# --- 3. Apply Mean Shift Filtering ---
clustered_img = cv2.pyrMeanShiftFiltering(
    img, 
    sp=spatial_radius, 
    sr=color_radius, 
    maxLevel=max_level
)

# --- 4. Display Results (BGR to RGB for Matplotlib) ---
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
clustered_rgb = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2RGB)

cv2.imwrite("testClusteredImage.png", clustered_rgb)


Z = img.reshape((-1, 3))
Z = np.float32(Z)
K = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(
    data=Z, 
    K=K, 
    bestLabels=None, 
    criteria=criteria, 
    attempts=10, 
    flags=cv2.KMEANS_RANDOM_CENTERS
)
center = np.uint8(center)
res = center[label.flatten()]
segmented_image = res.reshape((img.shape))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
segmented_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
cv2.imwrite("testKmeansImage.png", segmented_rgb)


cv2.imwrite("edgeboxes.png", edgeboxes(img))
cv2.imwrite("Kmeans-edgeboxes.png", edgeboxes(segmented_rgb))
cv2.imwrite("meanshift-edgeboxes.png", edgeboxes(clustered_rgb))

print(f"Loaded: {img_name}, Shape: {img.shape}")

