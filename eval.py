import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.ops import nms
import cv2
import numpy as np

from SelectiveSearch import selective_search_regions as selective_search

CONF_THRESH = 0.7
IOU_THRESH  = 0.3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE  = 160



def bbox_regression(boxes, deltas):
    px1, py1, px2, py2 = boxes.T
    pw  = px2 - px1
    ph  = py2 - py1
    pxc = px1 + 0.5 * pw
    pyc = py1 + 0.5 * ph

    dx, dy, dw, dh = deltas.T

    gx = pxc + dx * pw
    gy = pyc + dy * ph
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)

    rx1 = gx - 0.5 * gw
    ry1 = gy - 0.5 * gh
    rx2 = gx + 0.5 * gw
    ry2 = gy + 0.5 * gh

    return torch.stack([rx1, ry1, rx2, ry2], dim=1)


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


def compute_ap(preds, gts, iou_thresh=0.5):
    preds = sorted(preds, key=lambda x: x[1], reverse=True)

    tp = []
    fp = []
    matched = {}

    total_gt = sum(len(v) for v in gts.values())

    for img_id, score, box in preds:
        best_iou = 0
        best_gt  = -1

        for i, gt in enumerate(gts[img_id]):
            if (img_id, i) in matched:
                continue
            iou = compute_iou(box, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= iou_thresh:
            tp.append(1)
            fp.append(0)
            matched[(img_id, best_gt)] = True
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recalls = tp / (total_gt + 1e-6)
    precisions = tp / (tp + fp + 1e-6)

    ap = np.mean([
        max(precisions[recalls >= r], default=0)
        for r in np.linspace(0, 1, 11)
    ])

    return ap



def run_evaluation(model, dataloader):
    model.eval()

    all_predictions = []
    all_ground_truths = {}
    for img_id, batch in enumerate(dataloader):
        if batch is None:
            continue

        images = batch['image'].to(DEVICE)
        gt_boxes = batch['boxes'][0].numpy()
        all_ground_truths[img_id] = gt_boxes.tolist()

        img_cv = images[0].cpu().numpy().transpose(1,2,0)
        img_cv = (img_cv * 255).astype(np.uint8)
        img_cv = img_cv[:, :, ::-1].copy()

        prop_boxes, _ = selective_search(img_cv, num_regions=200)

        crops = []
        for box in prop_boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            crops.append(T.Resize((INPUT_SIZE, INPUT_SIZE))(images[0,:,y1:y2,x1:x2]))

        crops = torch.stack(crops).to(DEVICE)

        with torch.no_grad():
            logits, deltas = model(crops)
            probs = torch.softmax(logits, dim=1)[:,1]

        keep = probs > CONF_THRESH
        boxes  = prop_boxes[keep]
        deltas = deltas[keep]
        scores = probs[keep]

        boxes = bbox_regression(boxes, deltas)

        _, _, H, W = images.shape
        boxes[:,0::2].clamp_(0, W-1)
        boxes[:,1::2].clamp_(0, H-1)

        if len(boxes) > 0:
            keep_nms = nms(boxes, scores, IOU_THRESH)
            boxes  = boxes[keep_nms].cpu().numpy()
            scores = scores[keep_nms].cpu().numpy()

            for b, s in zip(boxes, scores):
                all_predictions.append((img_id, float(s), b.tolist()))
            img = images[0].cpu().numpy()      # (C, H, W)
        img = img.transpose(1, 2, 0)       # (H, W, C)
        img = (img * 255).astype("uint8")  # OpenCV expects uint8
        img = img[:, :, ::-1]              # RGB -> BGR
        img = img.copy()

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save annotated image every 50th sample
        if img_id % 50 == 0:
            out_name = f"{img_id:03d}_{filenames[0]}"
            cv2.imwrite(f"outputs/{out_name}", img)
            
    return all_predictions, all_ground_truths



from torch.utils.data.dataloader import default_collate

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


if __name__ == "__main__":
    from datasets import PotholeDataset
    from torch.utils.data import DataLoader
    from model import get_vgg16_model

    dataset = PotholeDataset(
        root_dir="/dtu/datasets1/02516/potholes",
        transform=T.Compose([T.Resize((INPUT_SIZE,INPUT_SIZE)), T.ToTensor()]),
        test=True,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=collate_skip_none)

    model = get_vgg16_model(pretrained=False, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load("models/final_model.pth", map_location=DEVICE))

    preds, gts = run_evaluation(model, dataloader)
    ap = compute_ap(preds, gts, IOU_THRESH)

    print(f" mAP@{IOU_THRESH}: {ap:.4f}")
