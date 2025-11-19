# Source - https://stackoverflow.com/a
# Posted by Nuzhny, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-19, License - CC BY-SA 4.0
# Script is changed to fit our needs

import cv2 as cv
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def edge_box(img):
    model = 'model.yml.gz'

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_img) / 255.0)

    kernel = np.ones((3, 3), np.uint8)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(30)
    boxes, scores = edge_boxes.getBoundingBoxes(edges, orimap)

    print(type(boxes), type(edges))
    return boxes, edges


if __name__ == '__main__':
    images = glob('/dtu/datasets1/02516/potholes/images/*.png')
    image = cv.imread(images[10])

    box_list, edge_image = edge_box(image)

    image_w_boxes = image.copy()
    for (x, y, w, h) in box_list:
        cv.rectangle(image_w_boxes, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Edges (Structured Forest)")
    plt.imshow(edge_image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("EdgeBoxes Proposals")
    plt.imshow(image_w_boxes)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

