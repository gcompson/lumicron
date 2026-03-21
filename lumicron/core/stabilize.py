import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def align_single_frame(f, frames_dir, out_dir, ref_img, warp_mode, criteria):
    """Worker function for parallel processing."""
    img = cv2.imread(os.path.join(frames_dir, f))
    if img is None:
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    try:
        _, warp_matrix = cv2.findTransformECC(ref_img, gray, warp_matrix, warp_mode, criteria)
        sz = (img.shape[1], img.shape[0])
        aligned = cv2.warpAffine(
            img, 
            warp_matrix, 
            sz, 
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        cv2.imwrite(os.path.join(out_dir, f), aligned)
    except (cv2.error, Exception):
        # Fallback to original if blurry
        cv2.imwrite(os.path.join(out_dir, f), img)
        
    return True

def stabilize_project(project_path):
    """
    Multi-core ECC Registration Engine.
    Nails the ground to a static coordinate system using ThreadPoolExecutor.
    """
    frames_dir = os.path.join(project_path, "02_FRAMES")
    out_dir = os.path.join(project_path, "02_STABILIZED")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not files:
        print("Error: No frames found in 02_FRAMES.")
        return

    ref_img = cv2.imread(os.path.join(frames_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        print(f"Error: Could not load reference frame {files[0]}")
        return

    warp_mode = cv2.MOTION_EUCLIDEAN
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-8)

    print(f"--- M4 MULTI-CORE STABILIZATION: {len(files)} FRAMES ---")
    
    # Unleash the M4 Pro: OpenCV releases the Python GIL during heavy C++ ops.
    # We use 8 workers to prevent thermal throttling while maximizing throughput.
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all frames to the thread pool
        futures = {
            executor.submit(
                align_single_frame, f, frames_dir, out_dir, ref_img, warp_mode, criteria
            ): f for f in files
        }
        
        # Wrap as_completed in tqdm for a smooth, single-line progress bar
        for future in tqdm(as_completed(futures), total=len(files), desc="Registering"):
            future.result() # Wait for individual thread to finish

    print(f"✅ Registration complete. Files saved to: {out_dir}")
