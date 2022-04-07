"""
Downloads and reads the dataset. 
"""

import os
import pandas as pd
import wget
import zipfile


"""
Creates a directory, if it does not exist.
Input: directory
Output: directory
"""
def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory : ", dir)
  else:
    print("Directory already existed : ", dir)
  return dir


"""
Downloads data from a specified url.
Input: url for dataset(str)
"""
def download_dataset_from_url(url):
  wget.download(url)


"""
Unzips dataset to a specified folder.
Input: path to unzip dataset(str), dataset filename
"""
def unzip_data_to_flder(path, filename):

  create_dir(path)
  
  with zipfile.ZipFile(filename, 'r') as zip_ref:
      zip_ref.extractall(path)
