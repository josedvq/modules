import os

def create_dir(base, new_folder):
    folder_path = os.path.join(base, new_folder)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path
