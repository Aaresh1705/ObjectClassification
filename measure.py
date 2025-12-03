from tkinter import Image
import torch
import numpy as np
import cv2
def pairwise_iou(gt_boxes, prop_boxes):
    """
    gt_boxes:   Tensor [N, 4]
    prop_boxes: Tensor [M, 4]

    Returns IoU matrix of size [N, M]
    This matrix can be interpreted as:
        iou_matrix[i, j] = IoU between gt_boxes[i] and prop_boxes[j]
    """
    if not isinstance(gt_boxes, torch.Tensor):
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
    if not isinstance(prop_boxes, torch.Tensor):
        prop_boxes = torch.tensor(prop_boxes, dtype=torch.float32)

    # Expand dims for broadcasting: [N,1,4] and [1,M,4]
    # We need to make the shapes match to use pytorch broadcasting
    A = gt_boxes[:, None, :]  # [N, 1, 4]
    B = prop_boxes[None, :, :]  # [1, M, 4]

    # Intersection coordinates
    # Finding maximum of xmins and ymins, minimum of xmaxs and ymaxs
    xA = torch.maximum(A[:,:, 0], B[:,:, 0])
    yA = torch.maximum(A[:,:, 1], B[:,:, 1])
    xB = torch.minimum(A[:,:, 2], B[:,:, 2])
    yB = torch.minimum(A[:,:, 3], B[:,:, 3])

    interW = (xB - xA).clamp(min=0)
    interH = (yB - yA).clamp(min=0)
    inter_area = interW * interH

    # Area of each GT and proposal
    areaA = (A[:,:, 2] - A[:,:, 0]) * (A[:,:, 3] - A[:,:, 1])
    areaB = (B[:,:, 2] - B[:,:, 0]) * (B[:,:, 3] - B[:,:, 1])

    # Union
    union = areaA + areaB - inter_area

    return inter_area / union.clamp(min=1e-6)

from PIL import ImageDraw, ImageFont

def visualize_best_proposals(image, gt_boxes, proposals, iou_threshold=0.0):
    """
    image:      PIL Image
    gt_boxes:   tensor/list of ground-truth boxes [N,4]
    proposals:  tensor/list of proposal boxes [M,4]

    Returns: PIL image with GT boxes (green) and best proposal (red) + IoU label
    """
    import torch

    # Ensure tensors
    if not isinstance(gt_boxes, torch.Tensor):
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
    if not isinstance(proposals, torch.Tensor):
        proposals = torch.tensor(proposals, dtype=torch.float32)

    # Compute pairwise IoUs:  [N, M]
    ious = pairwise_iou(gt_boxes, proposals)

    # Copy image for drawing
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Try loading a font (fallback to default if not found)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Colors
    gt_color = "green"
    best_color = "red"

    for i, gt in enumerate(gt_boxes):
        gt = gt.tolist()

        # Draw GT box
        draw.rectangle(gt, outline=gt_color, width=3)

        # Find best proposal for this GT box
        best_idx = torch.argmax(ious[i]).item()
        best_iou = ious[i, best_idx].item()
        best_box = proposals[best_idx].tolist()

        # Skip very low IoU matches if desired
        if best_iou <= iou_threshold:
            continue

        # Draw best proposal box
        draw.rectangle(best_box, outline=best_color, width=3)

        # Annotate IoU
        text = f"IoU={best_iou:.2f}"
        text_x, text_y = gt[0], gt[1] - 15
        draw.text((text_x, text_y), text, fill=best_color, font=font)
        
    #Annotate which color is whhat with white background
    legend_bg_height = 50
    legend_bg_width = 100
    draw.rectangle([0, 0, legend_bg_width, legend_bg_height], fill="white")
    legend_y = 10
    draw.text((10, legend_y), "GT Box", fill=gt_color, font=font)
    legend_y += 20
    draw.text((10, legend_y), "Best Proposal", fill=best_color, font=font)
    legend_y += 20

    return img



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from datasets import PotholeDataset
    loader = DataLoader(PotholeDataset(), batch_size=4, shuffle=True, collate_fn=lambda x: x)
    
    batch = next(iter(loader))
    from SelectiveSearch import selective_search_regions, edgeboxes
    for i, sample in enumerate(batch):
        dataset = PotholeDataset()
        image = sample["image"]

        gt_boxes = sample["boxes"]
        print(f"image {image}, gt_boxes {gt_boxes}")
        
        #change to numpy fro PIL image
        image = image.convert("RGB")
        image_np = np.array(image)
        proposals,image_with_boxes = selective_search_regions(image_np)  # or edgeboxes(image_np)
        # crop image using the proposals
        cv2.imwrite(f"images/selective_search_output_{i}.png", image_with_boxes)
        visualize_best_proposals(image, gt_boxes, proposals, iou_threshold=0.1).save(f"images/visualized_proposals_{i}.png")
    
    
    
