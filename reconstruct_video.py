import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool, cpu_count
from PIL import Image
import imagehash
from tqdm import tqdm
import os
import heapq