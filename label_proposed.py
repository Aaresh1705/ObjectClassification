import xml.etree.ElementTree as ET
import os
import cv2 as cv


def save_boxes_to_xml(image_path, box_list, output_dir="edge_boxes"):
    """
    Save bounding boxes to a Pascal VOC XML file.
    box_list = list of (x1, y1, x2, y2)
    """

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    folder = os.path.basename(os.path.dirname(image_path))
    xml_filename = os.path.splitext(filename)[0] + ".xml"
    xml_path = os.path.join(output_dir, xml_filename)

    # Load image to get size
    img = cv.imread(image_path)
    height, width, depth = img.shape

    # Root
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename

    # size
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    # Add objects
    for (x1, y1, x2, y2) in box_list:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "pothole"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "occluded").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(x1))
        ET.SubElement(bndbox, "ymin").text = str(int(y1))
        ET.SubElement(bndbox, "xmax").text = str(int(x2))
        ET.SubElement(bndbox, "ymax").text = str(int(y2))

    # Write file
    tree = ET.ElementTree(annotation)
    tree.write(xml_path)

    return xml_path

if __name__ == '__main__':
    pass