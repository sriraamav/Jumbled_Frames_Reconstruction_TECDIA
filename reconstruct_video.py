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
K_PHASH_CANDIDATES = 80
K_NEIGHBORS = 8  
BEAM_WIDTH = 15   
SSIM_THRESHOLD = 0.4  
NUM_WORKERS = max(1, cpu_count()-1)



def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[I] Extracted {len(frames)} frames")
    return frames, fps

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


def reconstruct_sequence(edges, start=None, beam_width=10, max_steps=2000):

    n = len(edges)
    if n == 0:
        return []

    # choose start node: one with highest degree (most neighbors)
    if start is None:
        start = max(range(n), key=lambda i: len(edges[i]))

    best_seq = None
    beam = [(0.0, [start], {start})]
    step = 0

    while beam and step < max_steps:
        step += 1
        new_beam = []

        for score, seq, visited in beam:
            last = seq[-1]
            expanded = False
            for s, nb in edges[last]:
                if nb in visited:
                    continue
                new_seq = seq + [nb]
                new_score = score - s
                new_visited = visited | {nb}
                new_beam.append((new_score, new_seq, new_visited))
                expanded = True

            # if no expansion possible but not yet all frames, we’ll fill later
            if not expanded and len(seq) == n:
                best_seq = seq
                break

        if best_seq:
            break

        if not new_beam:
            # completely stuck — pick the unvisited node most similar to the last frame
            current_tail = beam[0][1][-1]
            visited = beam[0][2]
            remaining = [i for i in range(n) if i not in visited]
            if not remaining:
                best_seq = beam[0][1]
                break
            # choose next by best available edge similarity
            candidate_scores = []
            for j in remaining:
                max_s = 0
                for i in visited:
                    for s, nb in edges[i]:
                        if nb == j and s > max_s:
                            max_s = s
                candidate_scores.append((max_s, j))
            if not candidate_scores:
                best_seq = beam[0][1] + remaining
                break
            best_next = max(candidate_scores)[1]
            new_seq = beam[0][1] + [best_next]
            new_beam = [(beam[0][0], new_seq, visited | {best_next})]

        # keep best few beams only
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

        # safety stop if sequence already covers all frames
        if any(len(seq) == n for _, seq, _ in beam):
            best_seq = max(beam, key=lambda x: len(x[1]))[1]
            break

    # if loop ended without full coverage, append any remaining frames
    if best_seq is None:
        best_seq = max(beam, key=lambda x: len(x[1]))[1]
    remaining = [i for i in range(n) if i not in best_seq]
    best_seq += remaining
    print(f"[I] Beam search finished in {step} steps. Sequence length = {len(best_seq)}")
    return best_seq



def local_refinement(sequence, grays, window=10):
    n = len(sequence)
    improved = True
    iter_count = 0
    while improved and iter_count < 2:
        improved = False
        iter_count += 1
        for start in range(0, n, window):
            end = min(n, start + window)
            best_seg = sequence[start:end]
            best_score = seg_score(best_seg, grays)

            for i in range(start, end):
                for j in range(i+1, end):
                    new_seg = sequence[start:end].copy()
                    li, lj = i-start, j-start
                    new_seg[li], new_seg[lj] = new_seg[lj], new_seg[li]
                    new_seq = sequence[:start] + new_seg + sequence[end:]
                    s = seg_score(new_seg, grays)
                    if s > best_score:
                        best_score = s
                        sequence = sequence[:start] + new_seg + sequence[end:]
                        improved = True
        if not improved:
            break
    return sequence

def fast_local_refinement(sequence, grays, passes=1):
    """Light heuristic pass to fix local discontinuities."""
    n = len(sequence)
    for _ in range(passes):
        i = 0
        while i < n - 2:
            a, b, c = sequence[i], sequence[i+1], sequence[i+2]
            s1 = ssim(grays[a], grays[b], data_range=grays[b].max()-grays[b].min())
            s2 = ssim(grays[b], grays[c], data_range=grays[c].max()-grays[c].min())
            if s1 < 0.3 and s2 < 0.3:   # discontinuity → try swapping
                sequence[i+1], sequence[i+2] = sequence[i+2], sequence[i+1]
                i = max(i-1, 0)          # re-check previous after swap
            else:
                i += 1
    return sequence


def seg_score(seg, grays):

    s = 0.0
    for a,b in zip(seg, seg[1:]):
        s += ssim(grays[a], grays[b], data_range=grays[b].max()-grays[b].min())
    return s

def write_video(original_frames, sequence, out_path, fps=30):

    h, w = original_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for idx in sequence:
        writer.write(original_frames[idx])
    writer.release()
    print(f"[I] Wrote reconstructed video to {out_path}")


def main(video_path, out_path):
    frames, fps = extract_frames(video_path)
    if len(frames) == 0:
        raise RuntimeError("No frames found")
    hashes, grays = phash_for_frames(frames)
    print("[I] Computing pHash shortlist candidates...")
    cand_idxs = shortlist_candidates(hashes, k=K_PHASH_CANDIDATES)
    print("[I] Computing SSIM graph (parallel)...")
    edges = compute_ssim_graph(grays, cand_idxs)
    print("[I] Reconstructing sequence (beam search)...")
    seq = reconstruct_sequence(edges)

    def sequence_smoothness(seq, grays):
        return sum(
            ssim(grays[a], grays[b], data_range=grays[b].max() - grays[b].min())
            for a, b in zip(seq, seq[1:])
        )

    smooth_forward  = sequence_smoothness(seq, grays)
    smooth_backward = sequence_smoothness(list(reversed(seq)), grays)

    if smooth_backward > smooth_forward:
        seq.reverse()
        print("[I] Sequence reversed for correct temporal direction.")

    print(f"[I] Final sequence length: {len(seq)} (should equal {len(frames)})")


    print("[I] Doing fast local refinement...")
    seq = fast_local_refinement(seq, grays, passes=2)
    print("[I] Writing video...")
    write_video(frames, seq, out_path, fps=fps)
    print("[I] Done.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python reconstruct_video.py scrambled.mp4 reconstructed.mp4")
        sys.exit(1)
    video_path = sys.argv[1]
    out_path = sys.argv[2]
    main(video_path, out_path)