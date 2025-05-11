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
    screenshot_map = {}  # Map HTML filenames to their corresponding screenshot paths
    tier_output_dir = os.path.join(output_root, tier_name)
    os.makedirs(tier_output_dir, exist_ok=True)

    # Configure Chrome in headless mode (no UI displayed)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920x1080")  # Consistent rendering size( for all screenshots)
    driver = webdriver.Chrome(options=chrome_options)

    # Step 1–2: Render each HTML file and take a screenshot
    for filename in sorted(os.listdir(tier_path)):
        if filename.endswith(".html"):
            filepath = os.path.abspath(os.path.join(tier_path, filename))
            screenshot_path = os.path.join(tier_output_dir, filename + ".png")
            try:
                driver.get("file://" + filepath)  # Open the HTML file in the browser
                driver.save_screenshot(screenshot_path)  # Save a screenshot of the rendered page
                screenshot_map[filename] = screenshot_path
            except Exception as e:
                print(f"Error rendering {filename}: {e}")
    driver.quit()  # Close the browser once all screenshots are taken

    # Step 3: Compute perceptual hashes (both pHash and dHash) for each screenshot
    hashes = []
    filenames = []
    for filename, path in screenshot_map.items():
        try:
            img = Image.open(path).convert("L").resize((256, 256))  # Convert to grayscale and resize
            ph = imagehash.phash(img)  # Perceptual hash (captures global image characteristics)
            dh = imagehash.dhash(img)  # Difference hash (captures local structure/edges)
            combined = np.concatenate((ph.hash.flatten(), dh.hash.flatten()))  # Combine both hashes
            hashes.append(combined)
            filenames.append(filename)
        except Exception as e:
            print(f"Hash error for {filename}: {e}")

    if not hashes:
        print("No valid screenshots.")
        return []

    hashes = np.array(hashes).astype(int)

    # Step 4: Cluster the screenshots using DBSCAN based on Hamming distance between hashes
    clustering = DBSCAN(eps=0.25, min_samples=2, metric='hamming')
    labels = clustering.fit_predict(hashes)

    # Step 5: Group filenames by cluster label
    groups = {}
    for label, filename in zip(labels, filenames):
        groups.setdefault(label, []).append(filename)

    final_groups = []
    for i, (label, files) in enumerate(groups.items()):
        # Optional: Save screenshots into folders grouped by visual similarity
        group_dir = os.path.join(tier_output_dir, f"group_{i if label != -1 else 'noise'}")
        os.makedirs(group_dir, exist_ok=True)
        for file in files:
            shutil.copy(screenshot_map[file], os.path.join(group_dir, file + ".png"))

        final_groups.append(files)

    return final_groups


# Main entry point
if __name__ == "__main__":
    ROOT_DIR = "./clones"  # Root directory containing HTML datasets organized by tier
    OUTPUT_DIR = "./visual_groups"  # Directory to store screenshots and/or grouped output

    all_results = {}
    # Iterate through each subdirectory in ROOT_DIR (each tier)
    for tier_folder in sorted(os.listdir(ROOT_DIR)):
        tier_path = os.path.join(ROOT_DIR, tier_folder)
        if os.path.isdir(tier_path):
            groups = process_tier(tier_path, tier_folder, OUTPUT_DIR)
            all_results[tier_folder] = groups

            # Print the grouped HTML files that are visually similar
            print(f"\nGroups for {tier_folder}:")
            for g in groups:
                print(g)
