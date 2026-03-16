import os
import shutil
from tqdm import tqdm

def copy_files_to_train():
    # Define source directories and destination directory
    source_dirs = ['./belge_Annotations', './ekar_Annotations']
    dest_dir = './train'

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory: {dest_dir}")

    total_files_copied = 0

    for source_dir in source_dirs:
        # Check if source directory exists
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory '{source_dir}' does not exist. Skipping.")
            continue

        print(f"Processing directory: {source_dir}")
        
        # Get list of files in the current source directory
        try:
            files = os.listdir(source_dir)
        except OSError as e:
            print(f"Error accessing {source_dir}: {e}")
            continue
        
        for filename in tqdm(files, desc=f"Copying from {source_dir}"):
            src_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)

            # Ensure we are copying a file, not a directory
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dest_file)
                total_files_copied += 1

    print(f"\nOperation complete. Copied {total_files_copied} files to '{dest_dir}'.")

if __name__ == "__main__":
    copy_files_to_train()
