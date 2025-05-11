import os
import shutil
import time
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import imagehash
from sklearn.cluster import DBSCAN
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor


# Set up a headless Chrome browser so we can render pages without opening windows
def create_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")       # run without GUI
    chrome_options.add_argument("--disable-gpu")    # avoid GPU overhead
    chrome_options.add_argument("--no-sandbox")     # for Docker compatibility
    chrome_options.add_argument("--window-size=1920x1080")  # consistent screenshots
    return webdriver.Chrome(options=chrome_options)


# This function renders the HTML file and takes a screenshot of it
# Optimization: This is run in parallel using multithreading
def render_and_screenshot_task(args):
    html_file_path, screenshot_path = args
    driver = None
    try:
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
        driver = create_chrome_driver()
        driver.get("file://" + html_file_path)
        time.sleep(0.2)  # give the page some time to fully load
        driver.save_screenshot(screenshot_path)
        return os.path.basename(html_file_path), screenshot_path
    except Exception as e:
        print(f"Rendering failed for {html_file_path}: {e}")
        return None
    finally:
        if driver:
            driver.quit()


# Computes two types of hashes for each screenshot: perceptual and difference
# Optimization: Uses threads instead of processes — image hashing is I/O + CPU-bound
def compute_hashes(screenshot_items):
    def _hash_item(item):
        filename, path = item
        try:
            img = Image.open(path).convert("L").resize((256, 256))  # standard size + grayscale
            ph = imagehash.phash(img)
            dh = imagehash.dhash(img)
            combined = np.concatenate((ph.hash.flatten(), dh.hash.flatten()))
            return filename, combined
        except Exception as e:
            print(f"Hash error for {filename}: {e}")
            return None

    # Thread pool is fine here since PIL operations are not heavy enough to need multiprocessing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_hash_item, screenshot_items))
    return [r for r in results if r is not None]


# Handles a folder of HTML files (called "tier") — renders, hashes, and clusters them
def process_tier(tier_path, tier_name, output_root, save_groups=False, max_workers=4):
    print(f"\nProcessing {tier_name}")

    # If we're saving groups, keep them under output_root; otherwise use /tmp
    if save_groups:
        tier_output_dir = os.path.join(output_root, tier_name)
    else:
        tier_output_dir = os.path.join("/tmp", f"temp_{tier_name}")
    os.makedirs(tier_output_dir, exist_ok=True)

    # Collect all HTML files to process
    html_files = sorted(f for f in os.listdir(tier_path) if f.endswith(".html"))
    tasks = []
    for filename in html_files:
        full_path = os.path.abspath(os.path.join(tier_path, filename))
        screenshot_path = os.path.join(tier_output_dir, filename + ".png")
        tasks.append((full_path, screenshot_path))

    # Optimization: parallel rendering using multiprocessing Pool
    print(f"Rendering {len(tasks)} HTML files using {max_workers} Chrome instances...")
    t0 = time.time()
    with Pool(processes=max_workers) as pool:
        results = pool.map(render_and_screenshot_task, tasks)
    print(f"Finished rendering in {time.time() - t0:.2f} seconds.")

    # Map screenshot results to their file names
    screenshot_map = {
        filename: path for result in results if result for filename, path in [result]
    }

    if not screenshot_map:
        print("No screenshots were successfully rendered.")
        return []

    # Optimization: compute hashes in parallel using threads
    hashed = compute_hashes(list(screenshot_map.items()))
    if not hashed:
        print("No valid hashes.")
        return []

    # Extract just the filename list and hash vectors
    filenames, hashes = zip(*hashed)
    hashes = np.array(hashes).astype(int)

    # Cluster the hash vectors using DBSCAN (no need to predefine # of clusters)
    # Using Hamming distance because the hashes are binary
    clustering = DBSCAN(eps=0.25, min_samples=2, metric='hamming')
    labels = clustering.fit_predict(hashes)

    # Group files by cluster label
    groups = {}
    for label, filename in zip(labels, filenames):
        groups.setdefault(label, []).append(filename)

    final_groups = []
    for i, (label, files) in enumerate(groups.items()):
        final_groups.append(files)
        if save_groups:
            group_dir = os.path.join(tier_output_dir, f"group_{i if label != -1 else 'noise'}")
            os.makedirs(group_dir, exist_ok=True)
            for file in files:
                shutil.copy(screenshot_map[file], os.path.join(group_dir, file + ".png"))

    # Cleanup: delete temporary screenshots
    for path in screenshot_map.values():
        try:
            os.remove(path)
        except Exception as e:
            print(f"Could not delete {path}: {e}")

    # If we didn't want to save groups, delete the entire temp directory
    if not save_groups:
        try:
            shutil.rmtree(tier_output_dir)
        except Exception as e:
            print(f"Could not delete temp folder {tier_output_dir}: {e}")

    return final_groups


# === Main script ===
if __name__ == "__main__":
    ROOT_DIR = "./clones"
    OUTPUT_DIR = "./visual_groups"
    SAVE_GROUPS = False  # Set to True if you want to inspect results visually
    MAX_WORKERS = min(8, cpu_count())  # Cap the number of parallel Chrome instances

    # Clean the output dir if it exists and we’re not saving groups
    if not SAVE_GROUPS and os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"Deleted output folder: {OUTPUT_DIR}")
        except Exception as e:
            print(f"Could not delete {OUTPUT_DIR}: {e}")

    # Go through each subdirectory (tier) in the dataset
    all_results = {}
    for tier_folder in sorted(os.listdir(ROOT_DIR)):
        tier_path = os.path.join(ROOT_DIR, tier_folder)
        if os.path.isdir(tier_path):
            groups = process_tier(tier_path, tier_folder, OUTPUT_DIR,
                                  save_groups=SAVE_GROUPS, max_workers=MAX_WORKERS)
            all_results[tier_folder] = groups

            print(f"\nGroups for {tier_folder}:")
            for g in groups:
                print(g)
