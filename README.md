[README.md](https://github.com/user-attachments/files/23288933/README.md)
## Jumbled Frames Reconstruction using SSIM + pHash Similarity

This project reconstructs the **correct temporal sequence** of a scrambled video by comparing visual similarities between frames using perceptual hashing (pHash) and Structural Similarity Index (SSIM).
It uses a **hybrid heuristic + graph-based + beam search** approach to efficiently find the most likely frame order, even when the original sequence is unknown.

---

## Installation

### 1️. Clone the Repository

```bash
git clone https://github.com/sriraamav/Jumbled_Frames_Reconstruction_TECDIA.git
cd Jumbled_Frames_Reconstruction_TECDIA
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


##  How to Run

### Basic Usage

```bash
python reconstruct_video.py scrambled.mp4 reconstructed.mp4
```

**Arguments:**

* `scrambled.mp4` → input video whose frames are out of order
* `reconstructed.mp4` → output video with reconstructed frame order

The program will:

1. Extract frames from the input video
2. Compute perceptual hashes (pHash) to shortlist visually similar candidates
3. Compute SSIM (Structural Similarity) between shortlisted frames
4. Build a similarity graph and find the most probable temporal sequence using **beam search**
5. Optionally refine local frame order for smoother continuity
6. Save the reconstructed video
---

---

## Approach and Design Explanation

###  1. Core Idea

The algorithm assumes that **temporally adjacent frames** in a video are **visually similar**.
Hence, by comparing all frames for similarity, we can reconstruct the most likely order.

---

###  2. Algorithms and Techniques Used

#### Perceptual Hashing (pHash)

* Used to quickly estimate **visual similarity** between frames.
* Produces a compact hash that changes slightly for visually similar images.
* Efficiently filters candidate pairs for more expensive SSIM comparison.

#### Structural Similarity Index (SSIM)

* A more accurate measure comparing luminance, contrast, and structure.
* Used for **fine-grained similarity scoring** once candidates are shortlisted via pHash.

#### Similarity Graph Construction

* Each frame is treated as a **node**.
* High-SSIM pairs are **edges** connecting frames likely to be temporally adjacent.
* Edges store SSIM scores (weights).

#### Beam Search (Sequence Reconstruction)

* Explores the graph to find the sequence of frames that **maximizes total similarity**.
* Keeps a limited number of best candidate sequences (beam width) to balance **accuracy vs. performance**.

#### Local Refinement

* After obtaining a rough sequence, performs **local swaps** or heuristic adjustments to fix discontinuities.
* A lightweight version (`fast_local_refinement`) uses SSIM-based heuristics to fix misordered triples.

---

### 3. Why This Method Was Chosen

| Challenge                  | Design Choice        | Reason                                          |
| -------------------------- | -------------------- | ----------------------------------------------- |
| Frame similarity detection | pHash + SSIM         | pHash for speed, SSIM for accuracy              |
| Large search space         | Beam search          | Avoids factorial complexity of full permutation |
| Noise and scene variations | Local refinement     | Smooths discontinuities and jitter              |
| CPU efficiency             | Multiprocessing Pool | Parallel SSIM computation                       |

---

###  4. Key Design Considerations

#### **Accuracy**

* Combining pHash and SSIM provides both coarse and fine visual similarity.
* Local refinement ensures smooth frame-to-frame continuity.

#### **Performance**

* pHash reduces candidate pairs drastically before SSIM evaluation.
* SSIM computations are distributed across CPU cores using Python’s `multiprocessing.Pool`.

#### **Scalability**

* Adjustable constants like `K_PHASH_CANDIDATES`, `K_NEIGHBORS`, and `SSIM_THRESHOLD` allow tuning for different video sizes.

#### **Robustness**

* Automatically reverses sequence direction if the backward smoothness is higher (handles flipped sequences).

---

## Adjustable Parameters (inside the script)

| Parameter            | Description                                        | Default       |
| -------------------- | -------------------------------------------------- | ------------- |
| `DOWNSAMPLE_HW`      | Resize frames before analysis to speed up SSIM     | (540, 960)    |
| `K_PHASH_CANDIDATES` | How many candidates per frame from pHash shortlist | 80            |
| `K_NEIGHBORS`        | How many strong neighbors per node kept            | 8             |
| `BEAM_WIDTH`         | Beam search width                                  | 15            |
| `SSIM_THRESHOLD`     | Minimum SSIM for edge creation                     | 0.4           |
| `NUM_WORKERS`        | Parallel processes for SSIM                        | All cores - 1 |

---

##  Example Output

```
[I] Extracted 240 frames
[I] Computing pHash shortlist candidates...
[I] Computing SSIM graph (parallel)...
SSIM pairs: 100%|███████████████████████████| 9600/9600 [00:45<00:00, 211.99it/s]
[I] Reconstructing sequence (beam search)...
[I] Beam search finished in 317 steps. Sequence length = 240
[I] Doing fast local refinement...
[I] Writing video...
[I] Wrote reconstructed video to reconstructed.mp4
[I] Done.
```

---
#

**Sriraam A V**

Contact:
[sriraamav18@gmail.com](mailto:sriraamav18@gmail.com)

