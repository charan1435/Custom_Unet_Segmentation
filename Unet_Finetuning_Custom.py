
# Imports
import torch 
import torch.nn as nn  
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader  
import cv2 
import numpy as np 
from PIL import Image 
import os 
import glob 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, jaccard_score 
from sklearn.model_selection import train_test_split 
from tqdm import tqdm 
from albumentations.pytorch import ToTensorV2 
import albumentations as A 
import warnings 
import argparse 
import random 
warnings.filterwarnings('ignore') 

# Setting seed value to 42  
def set_seed(seed=42):
 """Set random seeds for reproducibility""" 
 random.seed(seed) 
 np.random.seed(seed) 
 torch.manual_seed(seed) 
 torch.cuda.manual_seed_all(seed) 
 torch.backends.cudnn.deterministic = True 
 torch.backends.cudnn.benchmark = False

# Identify Label Maps 
def parse_labelmap_with_colors(base_dir): 
 
 """ parse the CVAT labelmap file to extract class names and their RGB colors """   

 # find the file 
 labelmap_path = os.path.join(base_dir,'labelmap') 
 if not os.path.exists(labelmap_path):
  labelmap_path = os.path.join(base_dir, 'labelmap.txt')  

 if not os.path.exists(labelmap_path):
  print(f"Path not found in base directory:{base_dir}") 
  return  None, None  
 
 print(f" Reading labelmap from: {labelmap_path}")  

 class_labels = {}  # Maps: Index -> class_name 
 color_to_index = {} # Maps : (R,G,B)  -> Index
 index = 0
 
 with open(labelmap_path, 'r') as f:
  for line in f: 
   line = line.strip()  # clears line  

   # skip comments (lines starting with #) and empty lines 
   if line.startswith('#') or not line: 
    continue  
   
   #Parse format line class name : R, G, B values 
   parts = line.split(':')  

   if len(parts) >= 2:
    class_name = parts[0].strip()
    rgb_str = parts[1].strip()

    if rgb_str:
     try:
      # Put them into correct form like (37, 0, 212)  
      # map allows to apply int to all the list values 
      r, g, b = map(int, rgb_str.split(",")) 
      color_tuple = (r, g, b)   

      #Check if colour exist already 
      if color_tuple in color_to_index:
       existing = class_labels[color_to_index[color_tuple]] 
       print (f"Duplicate color RGB {r},{g},{b}") 
       print (f"Already existing {existing}, slipping {class_name}") 
       continue  
      
      #Store the mapping 
      class_labels[index] = class_name
      color_to_index[color_tuple] = index 
      index+=1
      
     except Exception as e:
      print(f"Could not parse {class_name} : {e}")

 print(f"Total class loaded : {len(class_labels)}") 

 return class_labels , color_to_index


 
  


