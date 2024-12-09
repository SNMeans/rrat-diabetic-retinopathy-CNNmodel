import os

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not already exist.
    
    Args:
    - directory_path (str): Path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

