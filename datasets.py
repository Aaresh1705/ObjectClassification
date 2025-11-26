import os
import glob
from random import sample
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from extract_metadata import read_content
from PIL import ImageDraw
from torchvision import transforms
class PotholeDataset(Dataset):
    def __init__(self, root_dir="/dtu/datasets1/02516/potholes", transform=None):
        """
        root_dir: the path containing 'images' and 'annotations'
        """
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        self.transform = transform

        # Collect all XML files
        self.annotation_files = sorted(glob.glob(os.path.join(self.ann_dir, "*.xml")))
        if len(self.annotation_files) == 0:
            raise RuntimeError("No XML annotation files found!")

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        xml_path = self.annotation_files[idx]

        # Read annotation
        filename, boxes = read_content(xml_path)

        # Image
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert("RGB")

        # Convert boxes to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)


        # Return sample
        sample = {
            "image": img,
            "boxes": boxes,
            "filename": filename
        }

        if self.transform:
            sample["image"] = self.transform(img)
        return sample
    
    
def draw_boxes(image, boxes, colors=None, width=3):
    """
    Draw bounding boxes on a PIL image.

    image:  PIL.Image
    boxes:  list or tensor of [xmin, ymin, xmax, ymax]
    colors: optional list of colors to cycle through
    width:  line width
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    if colors is None:
        colors = [
            "red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan",
            "magenta", "lime", "teal", "lavender", "brown", "beige", "maroon",
            "navy", "olive", "coral", "grey", "white", "black"
        ]

    # Ensure boxes are a list of lists
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()

    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        draw.rectangle(box, outline=color, width=width)

    return img

class PotholeCropDataset(Dataset):
    def __init__(self, root_dir="/dtu/datasets1/02516/potholes", transform=None):
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        self.transform = transform

        self.samples = []  # (img_path, box, filename)

        annotation_files = sorted(glob.glob(os.path.join(self.ann_dir, "*.xml")))
        if len(annotation_files) == 0:
            raise RuntimeError("No XML annotation files found!")

        for xml_path in annotation_files:
            filename, boxes = read_content(xml_path)
            img_path = os.path.join(self.img_dir, filename)

            for box in boxes:
                self.samples.append((img_path, box, filename))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box, filename = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        xmin, ymin, xmax, ymax = box
        crop = img.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            crop = self.transform(crop)

        return {
            "image": crop,                        # Tensor [3, H, W]
            "box": torch.tensor(box),             # original box
            "filename": filename
        }



root = "/dtu/datasets1/02516/potholes"

dataset = PotholeDataset(root_dir=root)

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)

if __name__ == "__main__":
    batch = next(iter(loader))
    
    crop_batch = next(iter(crop_loader))
    for i, sample in enumerate(crop_batch):
        print(sample["filename"], sample["box"].shape)
        sample["image"].save(f"images/crop_sample_{i}_{sample['filename']}")
        
    # for sample in batch:
    #     print(sample["filename"], sample["boxes"].shape)
    #     sample["image"].save(f"images/sample_{sample['filename']}")
    #     img_with_boxes = draw_boxes(sample["image"], sample["boxes"])
    #     img_with_boxes.save(f"images/sample_boxes_{sample['filename']}")
