import os
import json
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

def create_voc_xml(json_data, image_filename, output_xml_path):
    """Creates a Pascal VOC XML file from LabelMe JSON data."""
    annotation = ET.Element("annotation")
    
    ET.SubElement(annotation, "folder").text = "Annotations"
    ET.SubElement(annotation, "filename").text = image_filename
    
    # Source
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    
    # Size
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(json_data.get("imageWidth", 0))
    ET.SubElement(size, "height").text = str(json_data.get("imageHeight", 0))
    ET.SubElement(size, "depth").text = "3" 
    
    ET.SubElement(annotation, "segmented").text = "0"
    
    # Objects
    for shape in json_data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points")
        
        if not points:
            continue
            
        # Calculate bounding box from points
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        xmin = min(x_coords)
        ymin = min(y_coords)
        xmax = max(x_coords)
        ymax = max(y_coords)
        
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))
        
    # Write to file with indentation for readability (optional, but good for debugging)
    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="    ", level=0)
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)

def main():
    source_root = '../test/json_etiketler'
    dest_folder = '../test/xml_etiketler'
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created directory: {dest_folder}")

    # Collect all JSON files recursively
    json_files = []
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files in {source_root}...")
    
    count = 0
    for json_path in tqdm(json_files, desc="Converting"):
        try:
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            parent_dir = os.path.dirname(json_path)
            
            # Load JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Find corresponding image
            # 1. Try finding image with same basename in the same folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            image_path = None
            ext_found = ""
            
            for ext in image_extensions:
                probe = os.path.join(parent_dir, base_name + ext)
                if os.path.exists(probe):
                    image_path = probe
                    ext_found = ext
                    break
            
            if not image_path:
                print(f"Warning: Image for {json_path} not found.")
                continue
            
            # Define destination paths
            dest_image_path = os.path.join(dest_folder, base_name + ext_found)
            dest_xml_path = os.path.join(dest_folder, base_name + '.xml')
            
            # Copy image and create XML
            shutil.copy2(image_path, dest_image_path)
            create_voc_xml(json_data, base_name + ext_found, dest_xml_path)
            count += 1
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            
    print(f"\nCompleted. Processed {count} pairs.")

if __name__ == '__main__':
    main()
