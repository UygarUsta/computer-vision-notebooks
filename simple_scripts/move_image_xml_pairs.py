import os
import shutil
from tqdm import tqdm 

def move_image_and_xml_pairs(source_folder='./train', dest_folder='./Annotations'):
    """
    Moves image and .xml pairs from a source folder to a destination folder.

    Args:
        source_folder (str): The path to the folder containing the original files.
        dest_folder (str): The path to the folder where files will be moved.
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created directory: {dest_folder}")

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    # Get a list of all files in the source folder
    try:
        files = os.listdir(source_folder)
    except FileNotFoundError:
        print(f"Error: Source folder not found at '{source_folder}'")
        return

    moved_count = 0
    for filename in tqdm(files):
        file_basename, file_extension = os.path.splitext(filename)

        # Check if the file is an image
        if file_extension.lower() in image_extensions:
            image_path = os.path.join(source_folder, filename)
            xml_filename = file_basename + '.xml'
            xml_path = os.path.join(source_folder, xml_filename)

            # Check if the corresponding .xml file exists
            if os.path.exists(xml_path):
                try:
                    # Move the image file
                    shutil.move(image_path, os.path.join(dest_folder, filename))
                    # Move the .xml file
                    shutil.move(xml_path, os.path.join(dest_folder, xml_filename))
                    print(f"Moved: {filename} and {xml_filename}")
                    moved_count += 1
                except shutil.Error as e:
                    print(f"Error moving files for {filename}: {e}")
            else:
                print(f"Warning: XML file not found for {filename}. Skipping.")

    print(f"\nProcess complete. Moved {moved_count} image/XML pairs.")

if __name__ == '__main__':
    # Define the source and destination directories
    train_directory = './train'
    annotations_directory = './Annotations'

    # Run the function
    move_image_and_xml_pairs(train_directory, annotations_directory)
