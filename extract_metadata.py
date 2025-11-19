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

    for p in paths:
        name, boxes = read_content(os.path.join(path, p))
        print(f"File: {name}, Boxes: {boxes}")
