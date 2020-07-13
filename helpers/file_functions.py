import os
import shutil 
import subprocess
from random import sample 

def create_dir(base, new_folder):
    folder_path = os.path.join(base, new_folder)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def sample_files(orig_folder, dest_folder, num):
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    files = list_files(orig_folder)
    
    assert num <= len(files)
    sampled_files = sample(files, num)

    for f in sampled_files:
        fname = os.path.basename(f)
        dest = shutil.copyfile(f, os.path.join(dest_folder, fname))

    return len(sampled_files)

def list_filenames(path):
    return sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

def list_files(path):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

def get_last_line(path):
    line = subprocess.check_output(['tail', '-1', path])

def basename_ext(path, extension):
    filename = os.path.basename(path)
    chunks = filename.split('.')
    return '{:s}.{:s}'.format(chunks[0], extension)