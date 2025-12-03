

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.ops import nms

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from extract_metadata import read_content
from torchvision.ops import nms
from SelectiveSearch import selective_search_regions as selective_search
# ----------------------------
# Configuration
# ----------------------------

CONF_THRESH = 0.7
IOU_THRESH  = 0.5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE  = 160

def run_evaluation(model, dataloader):
    model.eval()
    for batch in dataloader:
        images = batch['image'].to(DEVICE)
        boxes = batch['boxes'].to(DEVICE)
        filenames = batch['filename']

        prop_boxes = selective_search(images.cpu().numpy()[0], num_regions=200)
        #crop the image based on prop_boxes
        crops = []
        for box in prop_boxes:
            x1, y1, x2, y2 = box
            crop = images[0, :, y1:y2, x1:x2]
            crop = T.Resize((INPUT_SIZE, INPUT_SIZE))(crop)
            crops.append(crop)
        #classify the crops
        crops = torch.stack(crops).to(DEVICE)
        with torch.no_grad():
            outputs = model(crops)
            probs = nn.Softmax(dim=1)(outputs)[:, 1]  #probability of pothole class
        
        
        #filter by CONF_THRESH
        keep_idxs = torch.where(probs > CONF_THRESH)[0]
        kept_boxes = [prop_boxes[i] for i in keep_idxs]
        kept_probs = probs[keep_idxs]
        #For all the kept boxes, apply NMS
        if len(kept_boxes) > 0:
            kept_boxes_tensor = torch.tensor(kept_boxes).to(DEVICE)
            nms_idxs = nms(kept_boxes_tensor, kept_probs, IOU_THRESH)
            final_boxes = kept_boxes_tensor[nms_idxs].cpu().numpy()
        else:
            final_boxes = np.array([])
        print(f"Image: {filenames[0]}, Final boxes after NMS: {final_boxes}")
        #Save the image with the boxes drawn
        img = images[0].cpu().numpy().transpose(1,2,0).copy()
        img = (img * 255).astype(np.uint8)
        for box in final_boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f"outputs/{filenames[0]}", img)
        

if __name__ == "__main__":
    from ObjectClassification.datasets import PotholeDataset
    from torch.utils.data import DataLoader
    from ObjectClassification.model import SimpleCNN
    import os
    # Load dataset
    root = "/dtu/datasets1/02516/potholes"
    dataset = PotholeDataset(root_dir=root, transform=T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
    ]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = SimpleCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))

    # Run evaluation
    run_evaluation(model, dataloader)