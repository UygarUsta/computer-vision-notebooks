import os
import shutil
import random
from tqdm import tqdm

def create_validation_set(train_dir='./train', valid_dir='./valid', etiketler_dir='./etiketler', samples_per_folder=2):
    """
    Moves files from train to valid based on selection from etiketler folders.
    Selects 'samples_per_folder' random samples from each subfolder in 'etiketler_dir'.
    """
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
        print(f"Created validation directory: {valid_dir}")

    if not os.path.exists(train_dir):
        print(f"Error: Train directory '{train_dir}' does not exist.")
        return

    if not os.path.exists(etiketler_dir):
        print(f"Error: Etiketler directory '{etiketler_dir}' does not exist.")
        return

    # 1. Scan folders in etiketler and select samples
    files_to_move_basenames = set()
    
    # Get list of directories inside etiketler
    try:
        subdirs = [d for d in os.listdir(etiketler_dir) if os.path.isdir(os.path.join(etiketler_dir, d))]
    except OSError as e:
        print(f"Error accessing {etiketler_dir}: {e}")
        return

    print(f"Found {len(subdirs)} folders in {etiketler_dir}. Selecting {samples_per_folder} samples from each...")

    for subdir in subdirs:
        subdir_path = os.path.join(etiketler_dir, subdir)
        # Looking for json files as identifiers, or any file if json not found
        files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.json')]
        
        if not files:
            # Fallback if no json, look for images
            files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
        if not files:
            print(f"Warning: No suitable files found in {subdir_path}")
            continue

        # Select random samples
        selected = random.sample(files, min(len(files), samples_per_folder))
        
        for f in selected:
            basename = os.path.splitext(f)[0]
            files_to_move_basenames.add(basename)

    print(f"Selected {len(files_to_move_basenames)} unique samples.")

    # 2. Move files from train to valid
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    moved_count = 0
    
    for basename in tqdm(list(files_to_move_basenames), desc="Moving files"):
        xml_filename = basename + '.xml'
        src_xml = os.path.join(train_dir, xml_filename)
        
        if not os.path.exists(src_xml):
            print(f"Warning: {xml_filename} not found in {train_dir}")
            continue
        
        shutil.move(src_xml, os.path.join(valid_dir, xml_filename))
        
        # Find and move corresponding Image
        for ext in image_extensions:
            img_name = basename + ext
            src_img = os.path.join(train_dir, img_name)
            if os.path.exists(src_img):
                shutil.move(src_img, os.path.join(valid_dir, img_name))
                moved_count += 1
                break

    print(f"Done. Moved {moved_count} pairs to {valid_dir}.")

if __name__ == "__main__":
    create_validation_set()
