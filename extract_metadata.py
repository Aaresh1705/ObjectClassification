import os
import xml.etree.ElementTree as ET

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    list_with_all_boxes = []

    for obj in root.findall('object'):
        bnd = obj.find('bndbox')
        xmin = int(bnd.find('xmin').text)
        ymin = int(bnd.find('ymin').text)
        xmax = int(bnd.find('xmax').text)
        ymax = int(bnd.find('ymax').text)

        list_with_all_boxes.append([xmin, ymin, xmax, ymax])

    return filename, list_with_all_boxes

if __name__ == "__main__":
    path = "/dtu/datasets1/02516/potholes/annotations/"

    paths = os.listdir(path)
    print(paths)

    # Store TWO smallest: [(value, box, filename), (value, box, filename)]
    minimum_widths = [(float("inf"), None, None), (float("inf"), None, None)]
    minimum_heights = [(float("inf"), None, None), (float("inf"), None, None)]

    true_boxes = []

    for p in paths:
        name, boxes = read_content(os.path.join(path, p))
        print(f"File: {name}, Boxes: {boxes}")

        for box in boxes:
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            # ---- WIDTH (top 2 smallest) ----
            if width < minimum_widths[0][0]:
                minimum_widths[1] = minimum_widths[0]
                minimum_widths[0] = (width, box, name)
                true_boxes.append([xmin, ymin, xmax, ymax])

            elif width < minimum_widths[1][0]:
                minimum_widths[1] = (width, box, name)

            # ---- HEIGHT (top 2 smallest) ----
            if height < minimum_heights[0][0]:
                minimum_heights[1] = minimum_heights[0]
                minimum_heights[0] = (height, box, name)

            elif height < minimum_heights[1][0]:
                minimum_heights[1] = (height, box, name)

    # ---- PRINT RESULTS ----
    print("\nTwo smallest widths:")
    for w in minimum_widths:
        print(w)

    print("\nTwo smallest heights:")
    for h in minimum_heights:
        print(h)

    # ---- DRAW IMAGE (smallest width image) ----
    from datasets import draw_boxes
    import cv2
    from PIL import Image

    img = cv2.imread("/dtu/datasets1/02516/potholes/images/potholes123.png")
    img = Image.fromarray(img)

    img2 = draw_boxes(img, true_boxes)
    img2.save("img.png")
