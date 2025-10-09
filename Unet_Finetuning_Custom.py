
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

def rgb_mask_to_indexed(rgb_mask, color_to_index):
 
 """Covert RGB colores segementation mask to index mask""" 
 h , w = rgb_mask.shape[:2]  #Outputs the (h,w) from (h,w,c)
 indexed_mask = np.zeros((h, w), dtype=np.uint8)   #(creates are matrics with 0 for h * w and allocates 1 byte size for each pixel)

 #for each class color , find all pixels with that color 
 for color_tuple , class_idx in color_to_index.items():
  r, g, b = color_tuple  

  # Find pixels matching this color 
  # Tolerance of + or - 5 after JPEG Compression 

  mask = (
   (np.abs(rgb_mask[:, :, 0]- r) <= 5) &  # Red 
   (np.abs(rgb_mask[:, :, 1]- g) <= 5) &  # Green  
   (np.abs(rgb_mask[:, :, 2]- b) <= 5)    # Blue

  ) 

  # Set those pixels to the class index
  indexed_mask[mask] = class_idx

 return indexed_mask


def load_voc_format_pairs(base_dir): 
 """ Find the annotated maks and their original image pairs""" 
 images_dir = os.path.join(base_dir,'JPEGImages') 
 masks_dir = os.path.join(base_dir, 'SegmentationClass') 

 print(f"Searching Directories \n Images: {images_dir} \n Masks : {masks_dir}")  

 #verify if the directory exist 

 if not os.path.exists(images_dir):
  raise ValueError(f"JPEGImages not found : {images_dir}") 
 if not os.path.exists(masks_dir):
  raise ValueError(f"Segmentation Class not found : {masks_dir}") 
 
 #Find all the annotated mask files 
 mask_files=[] 
 for ext in ['*.png', '*.jpg','*.PNG','*.JPG'] :
  found = glob.glob(os.path.join(masks_dir,ext)) 
  mask_files.extend(found) 

 print(f"Found {len(mask_files)} annotation masks") 

 # For each mask find the original images 

 valid_pairs = [] 
 missing_images = [] 

 print(f" \n Pairing masks with original images") 
 for mask_path in sorted(mask_files): 
  mask_filename = os.path.basename(mask_path) 
  # Get file name without extension 
  mask_basename = os.path.splitext(mask_filename)[0]  

  #Try for different extensions 
  possible_extensions = ['.jpg','.jpeg','JPG','.JPEG','.PNG'] 

  image_path = None 
  for ext in possible_extensions:
   candidate = os.path.join(images_dir,mask_basename + ext) 
   if os.path.exists(candidate):
    image_path = candidate 
    break  
   
   if image_path:
    valid_pairs.append((image_path,mask_path)) 
    print(f"found pair for : {mask_basename}") 

   else:
    missing_images.append(mask_filename)
    print(f"No image for {mask_filename}") 

 print(f"\n{'='*60}")  
 print(f"Sucessfully paired : {len(valid_pairs)}")
 print(f"\n{'='*60}")  

 if missing_images:
  print (f"Image count without annotation {len(missing_images)}") 
  for missing in missing_images[:5]:
   print(f"{missing}") 

 if len(valid_pairs) == 0 :
  raise ValueError("No valid pairs found") 

 return valid_pairs 
 


 
  
 



 
 
  


