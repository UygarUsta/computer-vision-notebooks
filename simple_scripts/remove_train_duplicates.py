import os

def remove_duplicates_from_train(train_dir='./train', valid_dir='./valid'):
    """
    Scans the valid folder and removes any files with matching names found in the train folder.
    This ensures the training set does not contain any images or XMLs used in validation.
    """
    if not os.path.exists(valid_dir):
        print(f"Error: Validation directory '{valid_dir}' does not exist.")
        return

    if not os.path.exists(train_dir):
        print(f"Error: Train directory '{train_dir}' does not exist.")
        return

    # Get all filenames present in the valid directory
    valid_files = set(os.listdir(valid_dir))
    
    removed_count = 0
    print(f"Scanning for duplicates between '{valid_dir}' and '{train_dir}'...")

    for filename in valid_files:
        train_file_path = os.path.join(train_dir, filename)
        
        # If the file also exists in train, remove it
        if os.path.exists(train_file_path):
            try:
                os.remove(train_file_path)
                removed_count += 1
            except OSError as e:
                print(f"Error removing {filename}: {e}")

    print(f"Operation complete.")
    print(f"Removed {removed_count} files from '{train_dir}' that were present in '{valid_dir}'.")

if __name__ == "__main__":
    remove_duplicates_from_train()