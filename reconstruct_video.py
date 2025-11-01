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
    if DOWNSAMPLE_HW:
        h, w = DOWNSAMPLE_HW
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return gray, pil

def phash_for_frames(frames):
    hashes = []
    grays = []
    pil_images = []
    for f in frames:
        gray, pil = preprocess_frame(f)
        grays.append(gray)
        pil_images.append(pil)
        hashes.append(imagehash.phash(pil))  
    return hashes, grays

def hamming_distance(a, b):
    return (a - b).hash.astype(bool).sum() if False else bin(int(str(a), 16) ^ int(str(b), 16)).count("1")

def phash_hamming(a, b):
    return a - b

def shortlist_candidates(hashes, k=K_PHASH_CANDIDATES):
    n = len(hashes)
    cand_idxs = [None]*n
    for i in range(n):
        dists = [(phash_hamming(hashes[i], hashes[j]), j) for j in range(n) if j!=i]
        dists.sort(key=lambda x: x[0])
        cand_idxs[i] = [j for (_, j) in dists[:k]]
    return cand_idxs

def _ssim_worker(args):
    i, j, gray_i, gray_j = args
    score = ssim(gray_i, gray_j, data_range=gray_j.max() - gray_j.min())
    return (i, j, score)

def compute_ssim_graph(grays, candidate_indices):
    n = len(grays)
    edges = {i: [] for i in range(n)}  
    tasks = []
    for i in range(n):
        for j in candidate_indices[i]:
            if i < j:  
                tasks.append((i, j, grays[i], grays[j]))
    with Pool(NUM_WORKERS) as p:
        for (i, j, score) in tqdm(p.imap_unordered(_ssim_worker, tasks), total=len(tasks), desc="SSIM pairs"):
            if score >= SSIM_THRESHOLD:
                edges[i].append((score, j))
                edges[j].append((score, i))

    for i in range(n):
        edges[i].sort(reverse=True, key=lambda x: x[0])
        edges[i] = edges[i][:K_NEIGHBORS]
    return edges

