import os
import glob
from random import sample
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from extract_metadata import read_content
from PIL import ImageDraw

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

    def filter_small_boxes(self, boxes_list, min_size=5):
        filtered_boxes = []
        
        for box in boxes_list:
            x_min, y_min, x_max, y_max = box
            
            # Calculate width and height
            width = x_max - x_min
            height = y_max - y_min
            
            # Check if both dimensions meet or exceed the minimum size
            if width >= min_size and height >= min_size:
                filtered_boxes.append(box)
                
        return filtered_boxes

    def __getitem__(self, idx):
        xml_path = self.annotation_files[idx]

        # Read annotation
        filename, boxes = read_content(xml_path)

        boxes = self.filter_small_boxes(boxes)

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


root = "/dtu/datasets1/02516/potholes"

dataset = PotholeDataset(root_dir=root)

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)

if __name__ == "__main__":
    batch = next(iter(loader))

    for sample in batch:
        print(sample["filename"], sample["boxes"].shape)
        sample["image"].save(f"images/sample_{sample['filename']}")
        img_with_boxes = draw_boxes(sample["image"], sample["boxes"])
        img_with_boxes.save(f"images/sample_boxes_{sample['filename']}")

    # minx = 600
    # miny=600
    # filename = ""
    # bbox = []
    # for batch in loader:
    #     for sample in batch:
    #         #print the size of the boxes.
    #         boxes = sample['boxes']
    #         file = sample['filename']
    
    #         for box in boxes:
    #             horizontal_diff = abs(box[2] - box[0])
            
    #             vertical_diff = abs(box[3] - box[1])
    #             if horizontal_diff < minx and horizontal_diff > 1:
    #                 minx = horizontal_diff
    #                 filename = file
    #                 bbox = boxes
    #             if vertical_diff < miny and vertical_diff > 1:
    #                 miny = vertical_diff
    #                 filename = file
    #                 bbox = boxes

    # img = Image.open('/dtu/datasets1/02516/potholes/images/'+filename)
    # img2 = draw_boxes(img,bbox)
    # img2.save('image123.png')

    # print(minx,miny)    
    # print(filename)
