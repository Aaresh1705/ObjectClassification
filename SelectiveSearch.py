import cv2
import os
import random 
import numpy as np
import sys
import torch
import time

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
    boxes, s = edge_boxes.getBoundingBoxes(edges, orimap)

    for b in boxes:
        x, y, w, h = b
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

    return im

def selective_search_regions(img, num_regions=100, mode='fast'):
    """
    Apply Selective Search to an image and draw bounding boxes around proposed regions.
    
    Args:
        image_path (str): Path to the input image.
        num_regions (int): Number of top region proposals to draw.
        mode (str): 'fast' for faster processing, 'quality' for more proposals (slower).
    
    Returns:
        img_with_boxes (numpy.ndarray): Image with rectangles drawn.
    """
    # Load image
    # img = cv2.imread(image_path)
    
    # Create Selective Search segmentation object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    
    # Set mode
    if mode == 'fast':
        ss.switchToSelectiveSearchFast()
    elif mode == 'quality':
        ss.switchToSelectiveSearchQuality()
    else:
        raise ValueError("mode must be 'fast' or 'quality'")
    
    # Run selective search
    rects = ss.process()
    print(f"Total region proposals: {len(rects)}")
    
    # Draw top N region proposals
    # Draw boxes and convert to top-left/bottom-right
    img_with_boxes = img.copy()
    bboxes = []
    
    for i, (x, y, w, h) in enumerate(rects[:num_regions]):
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        bboxes.append([x1, y1, x2, y2])
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    # Convert to PyTorch tensor
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
    
    return bboxes_tensor, img_with_boxes

if __name__ == "__main__":
    # --- 1. Load Image ---
    folder = '/dtu/datasets1/02516/potholes/images'

    files = [f for f in os.listdir(folder) if f.endswith('.png')]
    img_name = random.choice(files)
    img_path = os.path.join(folder, img_name)

    img = cv2.imread('/dtu/datasets1/02516/potholes/images/potholes485.png')

    #cv2.imwrite("testimage.png", img)

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

    #cv2.imwrite("testClusteredImage.png", clustered_rgb)


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
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segmented_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("testKmeansImage.png", segmented_rgb)


    #cv2.imwrite("edgeboxes.png", edgeboxes(img))
    #cv2.imwrite("Kmeans-edgeboxes.png", edgeboxes(segmented_rgb))
    #cv2.imwrite("meanshift-edgeboxes.png", edgeboxes(clustered_rgb))
    t = time.time()
    a, v = selective_search_regions(img, num_regions=100, mode='fast')
    print(time.time() - t)
    cv2.imwrite("SelectiveSearch.png", v)



    print(f"Loaded: {img_name}, Shape: {img.shape}")

