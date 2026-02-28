from kagglehub import dataset_download
import shutil
import os

cache_path = dataset_download("ahmedxc4/skin-ds")
destination_folder = os.path.join(os.getcwd(), "dataset_raw")
if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)
shutil.copytree(cache_path, destination_folder)