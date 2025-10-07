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