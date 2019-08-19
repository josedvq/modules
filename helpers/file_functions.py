import os
import subprocess

def create_dir(base, new_folder):
    folder_path = os.path.join(base, new_folder)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def list_files(path):
	return sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

def get_last_line(path):
	line = subprocess.check_output(['tail', '-1', path])

def basename_ext(path, extension):
	filename = os.path.basename(path)
	chunks = filename.split('.')
	return '{:s}.{:s}'.format(chunks[0], extension)