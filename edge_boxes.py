# Source - https://stackoverflow.com/a
# Posted by Nuzhny, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-19, License - CC BY-SA 4.0
# Script is changed to fit our needs
import cv2
import cv2 as cv
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import time

def edge_boxes(img, num_regions):
    model = 'model.yml.gz'

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_img) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(num_regions)
    boxes, scores = edge_boxes.getBoundingBoxes(edges, orimap)

    boxes_xyxy = []

    for (x, y, w, h) in boxes:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        boxes_xyxy.append([x1, y1, x2, y2])

    return boxes_xyxy, edges

def visualize(t, edge_image, proposed_image):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Edges (Structured Forest)")
    plt.imshow(edge_image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("EdgeBoxes Proposals")
    plt.imshow(proposed_image)
    plt.axis("off")

    plt.title(t)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    images = glob('/dtu/datasets1/02516/potholes/images/potholes485.png')
    for image_file in images:
        image = cv2.imread(image_file)

        t1 = time.time()
        box_list, edge_image = edge_boxes(image.copy(), 100)
        print(time.time() - t1)

        image_w_boxes = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for (x1, y1, x2, y2) in box_list:
            cv.rectangle(image_w_boxes, (x1, y1), (x2, y2), (0, 255, 0), 1, cv.LINE_AA)

        visualize(image_file, edge_image, image_w_boxes)
