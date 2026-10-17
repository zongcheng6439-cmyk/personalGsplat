import cv2
import os
import argparse
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor

# Single image resize logic
def resize_task(args):
    img_path, save_path = args
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    resized = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(save_path, resized)

def fast_batch_resize(input_dir, output_dir, max_workers=12):
    os.makedirs(output_dir, exist_ok=True)
    task_queue = []

    # Pre-scan all valid images once
    for dirname in os.listdir(input_dir):
        print(f"prcess: {dirname}" )
        for filename in tqdm(os.listdir(os.path.join(input_dir, dirname))):
            ext = filename.lower()
            if ext.endswith((".jpg", ".jpeg", ".png")):
                full_in = os.path.join(input_dir, dirname, filename)
                full_out = os.path.join(output_dir, dirname, filename)
                os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)
                task_queue.append((full_in, full_out))

    # Parallel process all images
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(resize_task, task_queue)

if __name__ == "__main__":
    # Adjust max_workers: 8~16 for 8-core/16-core CPU
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest/")
    parser.add_argument("--image_dir", type=str, default="images") 
    parser.add_argument("--data_factor", type=int, default="2") 
    args = parser.parse_args()

    image_dir = os.path.join(args.dataset_dir, args.image_dir)
    output_dir = os.path.join(args.dataset_dir, f"images_{args.data_factor}_png")    

    fast_batch_resize(image_dir, output_dir, max_workers=12)
