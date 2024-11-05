import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2


class Converter():
    def __init__(self):
        pass
    def __call__(self,path,image_size,bboxes):
        annotation = self.create_pascal_voc_xml(path,image_size,bboxes)
        with open(path.split(".")[0]+".xml","w") as f:
            f.write(annotation)
    def create_pascal_voc_xml(self,image_name, image_size, bboxes):
        """
        Creates Pascal VOC XML annotation for an image.
        
        Parameters:
            image_name (str): Name of the image file (e.g., 'image1.jpg').
            image_size (tuple): (width, height, depth) of the image.
            bboxes (list): List of bounding boxes in the format [[xmin, ymin, xmax, ymax, class]].
        
        Returns:
            str: XML string in Pascal VOC format.
        """
        
        # Initialize the XML structure
        annotation = ET.Element("annotation")
        
        # Folder (optional, can be skipped or customized as needed)
        folder = ET.SubElement(annotation, "folder")
        folder.text = "images"

        # File name
        filename = ET.SubElement(annotation, "filename")
        filename.text = image_name

        # Size (image dimensions)
        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(image_size[0])
        height = ET.SubElement(size, "height")
        height.text = str(image_size[1])
        depth = ET.SubElement(size, "depth")
        depth.text = str(image_size[2])

        # Objects (bounding boxes)
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, obj_class = bbox
            
            obj = ET.SubElement(annotation, "object")
            name = ET.SubElement(obj, "name")
            name.text = obj_class
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            xmin_elem = ET.SubElement(bndbox, "xmin")
            xmin_elem.text = str(xmin)
            ymin_elem = ET.SubElement(bndbox, "ymin")
            ymin_elem.text = str(ymin)
            xmax_elem = ET.SubElement(bndbox, "xmax")
            xmax_elem.text = str(xmax)
            ymax_elem = ET.SubElement(bndbox, "ymax")
            ymax_elem.text = str(ymax)
        
        # Prettify XML
        xml_str = ET.tostring(annotation, encoding="utf-8")
        parsed_xml = minidom.parseString(xml_str)
        return parsed_xml.toprettyxml(indent="  ")


if __name__ == "__main__":
    converted = Converter()
    image = cv2.imread(r"C:\Users\PC\Desktop\images\honda_accord\17-honda-accord-2-4-i-vtec-prosmatec-2005-101573192.jpg")
    size = image.shape
    bbox = [[77,93,883,678,"car"],[416,517,569,598,"plate"]]
    anno = converted("17-honda-accord-2-4-i-vtec-prosmatec-2005-101573192.jpg",size,bbox)
