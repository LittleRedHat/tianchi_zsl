import os
import csv
import shutil

def normalize(image):
    for channel in range(image.size(0)):
        mean = image[channel,:,:].mean()
        std = image[channel,:,:].std()
        image[channel,:,:] = (image[channel,:,:] - mean) / std
    return image

def write_dict(s, file_path):
    w = csv.writer(open(file_path, "w"))
    for key, val in s.items():
        w.writerow([key, val])


def ensure_path(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def delete_path(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
