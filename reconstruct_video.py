import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool, cpu_count
from PIL import Image
import imagehash
from tqdm import tqdm
import os
import heapq

DOWNSAMPLE_HW = (540, 960)   
K_PHASH_CANDIDATES = 50    
K_NEIGHBORS = 5             
BEAM_WIDTH = 3               
SSIM_THRESHOLD = 0.7         
NUM_WORKERS = max(1, cpu_count()-1)



def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[I] Extracted {len(frames)} frames")
    return frames

def preprocess_frame(frame):
    # returns grayscale, optionally downsampled, and PIL for hashing
    if DOWNSAMPLE_HW:
        h, w = DOWNSAMPLE_HW
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return gray, pil
