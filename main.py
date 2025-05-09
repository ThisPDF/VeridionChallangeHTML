import os
import shutil
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import imagehash
from sklearn.cluster import DBSCAN
import numpy as np

def process_tier(tier_path, tier_name, output_root):
    print(f"\n Processing {tier_name}")
    screenshot_map = {}
    tier_output_dir = os.path.join(output_root, tier_name)
    os.makedirs(tier_output_dir, exist_ok=True)

    # Setup Chrome headless
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920x1080")
    driver = webdriver.Chrome(options=chrome_options)

    # Step 1–2: Take screenshots
    for filename in sorted(os.listdir(tier_path)):
        if filename.endswith(".html"):
            filepath = os.path.abspath(os.path.join(tier_path, filename))
            screenshot_path = os.path.join(tier_output_dir, filename + ".png")
            try:
                driver.get("file://" + filepath)
                driver.save_screenshot(screenshot_path)
                screenshot_map[filename] = screenshot_path
            except Exception as e:
                print(f"Error rendering {filename}: {e}")
    driver.quit()

    # Step 3: Compute combined perceptual hashes
    hashes = []
    filenames = []
    for filename, path in screenshot_map.items():
        try:
            img = Image.open(path).convert("L").resize((256, 256))
            ph = imagehash.phash(img)
            dh = imagehash.dhash(img)
            combined = np.concatenate((ph.hash.flatten(), dh.hash.flatten()))
            hashes.append(combined)
            filenames.append(filename)
        except Exception as e:
            print(f"Hash error for {filename}: {e}")

    if not hashes:
        print("No valid screenshots.")
        return []

    hashes = np.array(hashes).astype(int)

    # Step 4: Clustering (stricter tolerance)
    clustering = DBSCAN(eps=0.25, min_samples=2, metric='hamming')
    labels = clustering.fit_predict(hashes)

    # Step 5: Group files and copy screenshots
    groups = {}
    for label, filename in zip(labels, filenames):
        groups.setdefault(label, []).append(filename)

    final_groups = []
    for i, (label, files) in enumerate(groups.items()):
        group_dir = os.path.join(tier_output_dir, f"group_{i if label != -1 else 'noise'}")
        os.makedirs(group_dir, exist_ok=True)
        for file in files:
            shutil.copy(screenshot_map[file], os.path.join(group_dir, file + ".png"))
        final_groups.append(files)

    return final_groups

# === Run example ===
if __name__ == "__main__":
    ROOT_DIR = "./clones"  # e.g., "./clones_extracted/clones"
    OUTPUT_DIR = "./visual_groups"

    all_results = {}
    for tier_folder in sorted(os.listdir(ROOT_DIR)):
        tier_path = os.path.join(ROOT_DIR, tier_folder)
        if os.path.isdir(tier_path):
            groups = process_tier(tier_path, tier_folder, OUTPUT_DIR)
            all_results[tier_folder] = groups
            print(f"\n Output for {tier_folder}:")
            for g in groups:
                print(g)
