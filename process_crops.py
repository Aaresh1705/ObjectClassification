import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2
import os
import glob
import csv
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from extract_metadata import read_content
import torchvision.transforms as transforms
def pairwise_iou(gt_boxes, prop_boxes):
    A = gt_boxes[:, None, :]
    B = prop_boxes[None, :, :]

    xA = torch.max(A[..., 0], B[..., 0])
    yA = torch.max(A[..., 1], B[..., 1])
    xB = torch.min(A[..., 2], B[..., 2])
    yB = torch.min(A[..., 3], B[..., 3])

    inter = (xB - xA).clamp(0) * (yB - yA).clamp(0)

    areaA = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    areaB = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])

    union = areaA + areaB - inter
    return inter / union.clamp(min=1e-6)




def selective_search_boxes(image, max_regions=200):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()[:max_regions]

    return torch.tensor(
        [(x, y, x + w, y + h) for x, y, w, h in rects],
        dtype=torch.float32
    )
class SelectiveSearchCropDataset(Dataset):
    def __init__(
        self,
        root_dir,
        out_dir="ss_dataset",
        resize=(160, 160),
        max_regions=100,
        pos_iou=0.5,
        neg_per_gt=3
    ):
        self.root_dir = root_dir
        self.resize = resize
        self.max_regions = max_regions
        self.pos_iou = pos_iou
        self.neg_per_gt = neg_per_gt

        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")

        self.xmls = sorted(glob.glob(os.path.join(self.ann_dir, "*.xml")))

        self.img_out = os.path.join(out_dir, "images")
        os.makedirs(self.img_out, exist_ok=True)

        self.labels_path = os.path.join(out_dir, "labels.csv")

        # Write CSV header
        with open(self.labels_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "iou"])

    def __len__(self):
        return len(self.xmls)

    def __getitem__(self, idx):
        xml_path = self.xmls[idx]
        filename, gt_boxes = read_content(xml_path)
        #if filename ends with a number bigger than 458 skip it
        filename_no_ext = os.path.splitext(filename)[0]
        if int(''.join(filter(str.isdigit, filename_no_ext))) > 458:
            return None
        image_path = os.path.join(self.img_dir, filename)
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        num_gt = gt_boxes.shape[0]

        prop_boxes = selective_search_boxes(
            image_np, self.max_regions
        )

        ious = pairwise_iou(gt_boxes, prop_boxes)
        max_ious, _ = ious.max(dim=0)
        #save the dimensions of the closest gt box for each proposal
        closest_gt_boxes = gt_boxes[ious.argmax(dim=0)]
        
        labels = (max_ious >= self.pos_iou).int()

        pos_idx = torch.where(labels == 1)[0]
        neg_idx = torch.where(labels == 0)[0]

        # Background limit
        max_neg = self.neg_per_gt * num_gt
        if num_gt == 0:
            max_neg = 10

        if len(neg_idx) > max_neg:
            neg_idx = neg_idx[
                torch.randperm(len(neg_idx))[:max_neg]
            ]

        keep_idx = torch.cat([pos_idx, neg_idx])

        # Save
        with open(self.labels_path, "a", newline="") as f:
            writer = csv.writer(f)

            for j in keep_idx:
                box = prop_boxes[j]
                label = int(labels[j])
                iou = float(max_ious[j])

                x1, y1, x2, y2 = map(int, box.tolist())
                crop = image_pil.crop((x1, y1, x2, y2))
                # skip crops smaller than 5x5
                if crop.size[0] < 5 or crop.size[1] < 5:
                    continue
                crop = crop.resize(
                    self.resize,
                    Image.Resampling.BILINEAR
                )

                tag = "pos" if label == 1 else "neg"
                out_name = f"{idx:05d}_{j:04d}_{tag}.png"
                out_path = os.path.join(self.img_out, out_name)

                crop.save(out_path)
                # also write the box positions and iou to csv
                # also write the gt box positions and iou to csv
                writer.writerow([out_name, label, iou, x1, y1, x2, y2, closest_gt_boxes[j].tolist()])

        return None



from torch.utils.data import Dataset
import glob
from torchvision import transforms
class CropDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir/
            images/
                *.png
            labels.csv
        """
        self.labels_path = os.path.join(root_dir, "labels.csv")
        self.image_dir = os.path.join(root_dir, "images")

        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.samples = []
        self.gt_boxes = []
        self.proposed_boxes = []

        # Load CSV
        with open(self.labels_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(
                    (row["filename"], int(row["label"]))
                )
                self.gt_boxes.append([float(x) for x in row["gt_box"].strip("[]").split(",")])
                self.proposed_boxes.append([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])])
                
                

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]

        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        return image, label, torch.tensor(self.gt_boxes[idx]), torch.tensor(self.proposed_boxes[idx])

def cropDataLoader(batch_size=64, transform =None, train_ratio =0.7, val_ratio=0.15, test_ratio=0.15):
    if transform is None:
        transform =[]
    size = 160
    
    train_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        *transform,
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    total_len = len(CropDataset("./ss_crops"))
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len
    dataset = CropDataset("./ss_crops", transform=None)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader), (train_set, val_set, test_set)
    

if __name__ == "__main__":
    
    # # THIS RECREATES THE DATASET USING SELECTIVE SEARCH AND SAVES THE CROPS TO DISK
    # dataset = SelectiveSearchCropDataset(
    #     root_dir="/dtu/datasets1/02516/potholes",
    #     out_dir="ss_crops",
    #     pos_iou=0.5
    # )
    # for _ in dataset:
    #     pass  # generation side effects
    dataset = CropDataset("./ss_crops")

    (train_loader, val_loader, test_loader), (train_set, val_set, test_set) = cropDataLoader(batch_size=32)
    # import pdb;pdb.set_trace()
    img,label, gt_box, proposed_box = next(iter(train_loader))
    print(f"Dataset size: {len(dataset)}")

    img, label, gt_box, proposed_box = dataset[0]
    print(img, label, gt_box, proposed_box)